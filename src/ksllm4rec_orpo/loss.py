"""Memory-bounded sequence log-probabilities and stable ORPO loss."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class OrpoLossOutput:
    loss: torch.Tensor
    per_pair_loss: torch.Tensor
    sft_loss: torch.Tensor
    odds_ratio_loss: torch.Tensor
    log_odds: torch.Tensor


def reference_sequence_logps(
    logits: torch.Tensor, labels: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute average response log-probabilities from full logits."""

    if logits.ndim != 3 or labels.ndim != 2 or logits.shape[:2] != labels.shape:
        raise ValueError("Expected logits=(R,L,V) and labels=(R,L).")
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(-100)
    counts = valid.sum(dim=-1)
    if bool(counts.eq(0).any()):
        raise ValueError("Every response row must contain at least one valid label.")
    safe_labels = shift_labels.masked_fill(~valid, 0)
    token_logps = torch.gather(
        F.log_softmax(shift_logits, dim=-1),
        dim=-1,
        index=safe_labels.unsqueeze(-1),
    ).squeeze(-1)
    sums = (token_logps * valid).sum(dim=-1)
    return sums / counts, counts


def _chunk_ranges(size: int, chunk_size: int):
    for start in range(0, size, chunk_size):
        yield start, min(start + chunk_size, size)


class _ChunkedSequenceLogps(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        valid_hidden: torch.Tensor,
        lm_head_weight: torch.Tensor,
        targets: torch.Tensor,
        row_ids: torch.Tensor,
        token_counts: torch.Tensor,
        chunk_size: int,
    ) -> torch.Tensor:
        if valid_hidden.ndim != 2:
            raise ValueError("valid_hidden must have shape (N,D).")
        if targets.ndim != 1 or row_ids.ndim != 1:
            raise ValueError("targets and row_ids must be one-dimensional.")
        if valid_hidden.size(0) == 0:
            raise ValueError("At least one valid response token is required.")
        if lm_head_weight.requires_grad:
            raise ValueError("Chunked ORPO requires a frozen LM Head.")
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive.")
        if bool(token_counts.le(0).any()):
            raise ValueError("Every response row must contain a valid token.")

        ctx.save_for_backward(
            valid_hidden, lm_head_weight, targets, row_ids, token_counts
        )
        ctx.chunk_size = int(chunk_size)
        sums = torch.zeros(
            token_counts.numel(), dtype=torch.float32, device=valid_hidden.device
        )
        for start, end in _chunk_ranges(valid_hidden.size(0), ctx.chunk_size):
            logits = F.linear(valid_hidden[start:end], lm_head_weight).float()
            logps = -F.cross_entropy(logits, targets[start:end], reduction="none")
            sums.index_add_(0, row_ids[start:end], logps)
        return sums / token_counts.to(torch.float32)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        valid_hidden, lm_head_weight, targets, row_ids, token_counts = (
            ctx.saved_tensors
        )
        grad_hidden = torch.empty_like(valid_hidden)
        scales = grad_output.to(torch.float32) / token_counts.to(torch.float32)
        for start, end in _chunk_ranges(valid_hidden.size(0), ctx.chunk_size):
            with torch.enable_grad():
                hidden_chunk = valid_hidden[start:end].detach().requires_grad_(True)
                logits = F.linear(hidden_chunk, lm_head_weight).float()
                logps = -F.cross_entropy(
                    logits, targets[start:end], reduction="none"
                )
                weighted = (
                    logps * scales[row_ids[start:end]].to(logps.dtype)
                ).sum()
                (chunk_grad,) = torch.autograd.grad(weighted, hidden_chunk)
            grad_hidden[start:end].copy_(chunk_grad.to(grad_hidden.dtype))
        return grad_hidden, None, None, None, None, None


def chunked_sequence_logps(
    hidden_states: torch.Tensor,
    lm_head_weight: torch.Tensor,
    labels: torch.Tensor,
    *,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-row average response log-probabilities without full logits."""

    if hidden_states.ndim != 3 or labels.ndim != 2:
        raise ValueError("Expected hidden_states=(R,L,D), labels=(R,L).")
    if hidden_states.shape[:2] != labels.shape:
        raise ValueError("Hidden-state and label sequence shapes must match.")
    shift_hidden = hidden_states[:, :-1, :]
    shift_labels = labels[:, 1:]
    valid = shift_labels.ne(-100)
    token_counts = valid.sum(dim=-1)
    if bool(token_counts.eq(0).any()):
        raise ValueError("Every response row must contain at least one valid label.")
    row_grid = torch.arange(labels.size(0), device=labels.device).unsqueeze(1)
    row_ids = row_grid.expand_as(shift_labels)[valid]
    logps = _ChunkedSequenceLogps.apply(
        shift_hidden[valid],
        lm_head_weight,
        shift_labels[valid],
        row_ids,
        token_counts,
        chunk_size,
    )
    return logps, token_counts


def _log1mexp(log_probability: torch.Tensor) -> torch.Tensor:
    """Stable log(1-exp(x)) for x < 0."""

    x = log_probability.float().clamp_max(-torch.finfo(torch.float32).eps)
    cutoff = -math.log(2.0)
    return torch.where(
        x < cutoff,
        torch.log1p(-torch.exp(x)),
        torch.log(-torch.expm1(x)),
    )


def stable_orpo_loss(
    chosen_logps: torch.Tensor,
    rejected_logps: torch.Tensor,
    *,
    beta: float = 0.1,
) -> OrpoLossOutput:
    if chosen_logps.ndim != 1 or rejected_logps.shape != chosen_logps.shape:
        raise ValueError("chosen_logps and rejected_logps must share shape (B,).")
    if beta <= 0.0:
        raise ValueError("ORPO beta must be positive.")
    log_odds = (chosen_logps.float() - rejected_logps.float()) - (
        _log1mexp(chosen_logps) - _log1mexp(rejected_logps)
    )
    sft_loss = -chosen_logps.float()
    odds_ratio_loss = -F.logsigmoid(log_odds)
    per_pair_loss = sft_loss + float(beta) * odds_ratio_loss
    loss = per_pair_loss.mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("ORPO loss is NaN or Inf.")
    return OrpoLossOutput(
        loss=loss,
        per_pair_loss=per_pair_loss,
        sft_loss=sft_loss,
        odds_ratio_loss=odds_ratio_loss,
        log_odds=log_odds,
    )
