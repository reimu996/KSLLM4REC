"""Memory-bounded focal loss for a frozen language-model output head."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class LossMetrics:
    """Detached token statistics produced while evaluating one microbatch."""

    item_ratio: float
    item_loss: float | None
    text_loss: float | None
    dataset_loss: float
    valid_tokens: int
    item_tokens: int


def reference_focal_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    item_mask: torch.Tensor,
    *,
    gamma: float = 2.0,
    item_weight: float = 3.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Implement the approved submission loss literally for equivalence tests."""

    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()
    shift_item_mask = item_mask[..., 1:].contiguous()
    per_token_loss = F.cross_entropy(
        shift_logits.float().view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(shift_labels.shape)
    valid = shift_labels != -100
    if not bool(valid.any()):
        raise ValueError(
            "A microbatch must contain at least one valid assistant token."
        )

    p = torch.exp(-per_token_loss)
    focal = (1.0 - p).pow(gamma) * per_token_loss
    weights = torch.where(shift_item_mask, item_weight, 1.0)
    return (focal * weights)[valid].mean(), per_token_loss[valid].detach()


def _chunk_ranges(size: int, chunk_size: int) -> Iterable[tuple[int, int]]:
    for start in range(0, size, chunk_size):
        yield start, min(start + chunk_size, size)


def _token_losses(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    targets: torch.Tensor,
    item_flags: torch.Tensor,
    gamma: float,
    item_weight: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    logits = F.linear(hidden, lm_head_weight).float()
    ce = F.cross_entropy(logits, targets, reduction="none")
    p = torch.exp(-ce)
    focal = (1.0 - p).pow(gamma) * ce
    weights = torch.where(item_flags, item_weight, 1.0)
    return focal * weights, ce


class _ChunkedFocalLoss(torch.autograd.Function):
    """Discard output-head logits in forward and recompute them in backward."""

    @staticmethod
    def forward(
        ctx,
        valid_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        targets: torch.Tensor,
        item_flags: torch.Tensor,
        gamma: float,
        item_weight: float,
        chunk_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if valid_hidden.ndim != 2:
            raise ValueError(
                f"valid_hidden must be rank 2, got {tuple(valid_hidden.shape)}"
            )
        if valid_hidden.size(0) == 0:
            raise ValueError(
                "A microbatch must contain at least one valid assistant token."
            )
        if lm_head_weight.requires_grad:
            raise ValueError("The LM head must be frozen for chunked focal loss.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")

        ctx.save_for_backward(valid_hidden, lm_head_weight, targets, item_flags)
        ctx.gamma = float(gamma)
        ctx.item_weight = float(item_weight)
        ctx.chunk_size = int(chunk_size)

        loss_sum = torch.zeros((), dtype=torch.float32, device=valid_hidden.device)
        ce_chunks: list[torch.Tensor] = []
        for start, end in _chunk_ranges(valid_hidden.size(0), ctx.chunk_size):
            weighted_focal, ce = _token_losses(
                valid_hidden[start:end],
                lm_head_weight,
                targets[start:end],
                item_flags[start:end],
                ctx.gamma,
                ctx.item_weight,
            )
            loss_sum.add_(weighted_focal.sum())
            ce_chunks.append(ce.detach())

        ce_values = torch.cat(ce_chunks, dim=0)
        ctx.mark_non_differentiable(ce_values)
        return loss_sum / valid_hidden.size(0), ce_values

    @staticmethod
    def backward(ctx, grad_loss: torch.Tensor, grad_ce: torch.Tensor | None):
        valid_hidden, lm_head_weight, targets, item_flags = ctx.saved_tensors
        grad_hidden = torch.empty_like(valid_hidden)
        denominator = valid_hidden.size(0)

        for start, end in _chunk_ranges(denominator, ctx.chunk_size):
            with torch.enable_grad():
                hidden_chunk = valid_hidden[start:end].detach().requires_grad_(True)
                weighted_focal, _ = _token_losses(
                    hidden_chunk,
                    lm_head_weight,
                    targets[start:end],
                    item_flags[start:end],
                    ctx.gamma,
                    ctx.item_weight,
                )
                chunk_loss = weighted_focal.sum() / denominator
                (chunk_grad,) = torch.autograd.grad(
                    chunk_loss,
                    hidden_chunk,
                    grad_outputs=grad_loss.to(chunk_loss.dtype),
                )
            grad_hidden[start:end].copy_(chunk_grad.to(grad_hidden.dtype))

        return grad_hidden, None, None, None, None, None, None


def chunked_focal_loss(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    item_mask: torch.Tensor,
    *,
    gamma: float = 2.0,
    item_weight: float = 3.0,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute the approved loss without materializing full-sequence logits.

    Returns the differentiable scalar loss, detached CE values for valid tokens,
    and the corresponding detached item flags.
    """

    if hidden_states.ndim != 3 or labels.ndim != 2 or item_mask.ndim != 2:
        raise ValueError(
            "Expected hidden_states=(B,L,D), labels=(B,L), item_mask=(B,L)."
        )
    if hidden_states.shape[:2] != labels.shape or labels.shape != item_mask.shape:
        raise ValueError(
            "hidden_states, labels, and item_mask sequence shapes must match."
        )

    shift_hidden = hidden_states[..., :-1, :]
    shift_labels = labels[..., 1:]
    shift_items = item_mask[..., 1:]
    valid = shift_labels != -100
    valid_hidden = shift_hidden[valid]
    targets = shift_labels[valid]
    item_flags = shift_items[valid].bool()

    loss, ce_values = _ChunkedFocalLoss.apply(
        valid_hidden,
        lm_head_weight,
        targets,
        item_flags,
        gamma,
        item_weight,
        chunk_size,
    )
    return loss, ce_values, item_flags.detach()


def summarize_metrics(ce_values: torch.Tensor, item_flags: torch.Tensor) -> LossMetrics:
    """Match the submission.py metric definitions without recomputing CE."""

    if (
        ce_values.ndim != 1
        or item_flags.ndim != 1
        or ce_values.shape != item_flags.shape
    ):
        raise ValueError(
            "ce_values and item_flags must be aligned one-dimensional tensors."
        )
    valid_tokens = ce_values.numel()
    if valid_tokens == 0:
        raise ValueError("Cannot summarize an empty set of valid tokens.")

    item_flags = item_flags.bool()
    text_flags = ~item_flags
    item_tokens = int(item_flags.sum().item())
    item_loss = float(ce_values[item_flags].mean().item()) if item_tokens else None
    text_loss = (
        float(ce_values[text_flags].mean().item()) if bool(text_flags.any()) else None
    )
    return LossMetrics(
        item_ratio=item_tokens / valid_tokens,
        item_loss=item_loss,
        text_loss=text_loss,
        dataset_loss=float(ce_values.mean().item()),
        valid_tokens=valid_tokens,
        item_tokens=item_tokens,
    )
