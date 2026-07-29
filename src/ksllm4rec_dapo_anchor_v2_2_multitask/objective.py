"""V2.2 reward, clipped RL, and bounded GT-set anchor objectives.

The anchor target is the total probability mass assigned to every legal GT SID
for one prompt.  It is deliberately separate from the on-policy RLOO branch:
the caller measures its raw gradient and scales it before the one permitted
anchor optimizer update.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch

from ._infra.orpo_data import Sid
from ._infra.rloo_objective import sid_reward

from . import contract


@dataclass(frozen=True)
class RewardOutput:
    """Five-level rewards and centered group-relative advantages for G=16."""

    rewards: torch.Tensor
    advantages: torch.Tensor
    mean_reward: torch.Tensor
    std_reward: torch.Tensor
    tiers: tuple[str, ...]
    effective: bool


def is_effective_rewards(rewards: torch.Tensor) -> bool:
    if rewards.shape != (contract.GROUP_SIZE,):
        raise ValueError(f"rewards must have shape ({contract.GROUP_SIZE},).")
    values = rewards.detach().float()
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("rewards contain NaN or Inf.")
    return bool(values.max().item() > values.min().item())


def group_relative_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Keep the existing DAPO-anchor v1 advantage: ``r - mean(r)``."""

    if rewards.shape != (contract.GROUP_SIZE,):
        raise ValueError(f"rewards must have shape ({contract.GROUP_SIZE},).")
    values = rewards.float()
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("rewards contain NaN or Inf.")
    return values - values.mean()


def group_relative_rewards_and_advantages(
    candidates: Sequence[Sid], positives: Sequence[Sid]
) -> RewardOutput:
    live = tuple(candidates)
    if len(live) != contract.GROUP_SIZE:
        raise ValueError(f"Expected exactly {contract.GROUP_SIZE} candidates.")
    scored = [sid_reward(candidate, positives) for candidate in live]
    rewards = torch.tensor([value for value, _ in scored], dtype=torch.float32)
    return RewardOutput(
        rewards=rewards,
        advantages=group_relative_advantages(rewards),
        mean_reward=rewards.mean(),
        std_reward=rewards.std(correction=0),
        tiers=tuple(tier for _, tier in scored),
        effective=is_effective_rewards(rewards),
    )


@dataclass(frozen=True)
class ClippedChunkOutput:
    loss_sum: torch.Tensor
    per_candidate_loss_sums: torch.Tensor
    ratios: torch.Tensor
    clip_active: torch.Tensor
    below_low: torch.Tensor
    above_high: torch.Tensor
    decision_counts: torch.Tensor


def clipped_group_relative_token_sum(
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    decision_mask: torch.Tensor,
    *,
    clip_low: float = contract.CLIP_RATIO_LOW,
    clip_high: float = contract.CLIP_RATIO_HIGH,
) -> ClippedChunkOutput:
    """Return the existing asymmetric-clipped RL loss for one C<=8 chunk."""

    if new_logps.ndim != 2 or not 1 <= new_logps.shape[0] <= contract.LOSS_CHUNK_SIZE:
        raise ValueError("new_logps must have shape (C,T) with 1 <= C <= 8.")
    if old_logps.shape != new_logps.shape:
        raise ValueError("old_logps must match new_logps.")
    if decision_mask.shape != new_logps.shape or decision_mask.dtype != torch.bool:
        raise ValueError("decision_mask must be boolean and match logps.")
    if advantages.shape != (new_logps.shape[0],):
        raise ValueError("advantages must have shape (C,).")
    if not 0.0 < float(clip_low) < 1.0 < float(clip_high):
        raise ValueError("clip bounds must satisfy 0 < low < 1 < high.")
    if not bool(decision_mask.any()):
        raise ValueError("A clipped chunk needs at least one decision token.")

    new_values = new_logps.float()[decision_mask]
    old_values = old_logps.detach().float()[decision_mask]
    expanded_advantages = advantages.detach().float()[:, None].expand_as(new_logps)
    token_advantages = expanded_advantages[decision_mask]
    if not bool(torch.isfinite(new_values).all()) or not bool(
        torch.isfinite(old_values).all()
    ):
        raise FloatingPointError("log probabilities contain NaN or Inf.")
    if not bool(torch.isfinite(token_advantages).all()):
        raise FloatingPointError("advantages contain NaN or Inf.")

    ratios = torch.exp(new_values - old_values)
    clipped_ratios = ratios.clamp(min=float(clip_low), max=float(clip_high))
    unclipped_losses = -(ratios * token_advantages)
    clipped_losses = -(clipped_ratios * token_advantages)
    token_losses = torch.maximum(unclipped_losses, clipped_losses)
    if not bool(torch.isfinite(token_losses).all()):
        raise FloatingPointError("clipped token losses contain NaN or Inf.")

    loss_matrix = torch.zeros_like(new_logps, dtype=torch.float32)
    loss_matrix[decision_mask] = token_losses
    counts = decision_mask.sum(dim=-1)
    return ClippedChunkOutput(
        loss_sum=token_losses.sum(),
        per_candidate_loss_sums=loss_matrix.sum(dim=-1),
        ratios=ratios.detach(),
        clip_active=(clipped_losses > unclipped_losses).detach(),
        below_low=(ratios < float(clip_low)).detach(),
        above_high=(ratios > float(clip_high)).detach(),
        decision_counts=counts,
    )


@dataclass(frozen=True)
class AnchorSetLossOutput:
    """One prompt's GT-set probability-mass target.

    ``sequence_logps[m]`` is the sum of decision-token log probabilities for
    GT SID ``m``.  The final loss is ``-logsumexp(sequence_logps)``.
    """

    loss: torch.Tensor
    sequence_logps: torch.Tensor
    decision_counts: torch.Tensor


def gt_sequence_logps(
    logps: torch.Tensor, decision_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce ``(M,T)`` GT log-probabilities to one score per GT SID."""

    if logps.ndim != 2 or logps.shape[0] < 1:
        raise ValueError("GT logps must have shape (M,T) with M >= 1.")
    if decision_mask.shape != logps.shape or decision_mask.dtype != torch.bool:
        raise ValueError("GT decision_mask must be boolean and match logps.")
    if decision_mask.device != logps.device:
        raise ValueError("GT logps and decision_mask must share one device.")
    values = logps.float()[decision_mask]
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("GT logps contain NaN or Inf at decision tokens.")
    token_values = torch.zeros_like(logps, dtype=torch.float32)
    token_values[decision_mask] = values
    return token_values.sum(dim=-1), decision_mask.sum(dim=-1)


def gt_set_anchor_loss(sequence_logps: torch.Tensor) -> torch.Tensor:
    """Compute ``-log(sum_m P(GT_m | prompt))`` for one prompt's GT set."""

    if sequence_logps.ndim != 1 or sequence_logps.numel() < 1:
        raise ValueError("sequence_logps must have shape (M,) with M >= 1.")
    values = sequence_logps.float()
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("GT sequence_logps contain NaN or Inf.")
    loss = -torch.logsumexp(values, dim=0)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("GT-set anchor loss is NaN or Inf.")
    return loss


def gt_all_decision_nll(
    sequence_logps: torch.Tensor, decision_counts: torch.Tensor
) -> torch.Tensor:
    """Auxiliary, human-readable mean NLL across every GT decision token."""

    if sequence_logps.ndim != 1 or decision_counts.shape != sequence_logps.shape:
        raise ValueError("GT sequence scores and decision counts must both have shape (M,).")
    total = int(decision_counts.sum().item())
    if total <= 0:
        return torch.zeros((), dtype=torch.float32, device=sequence_logps.device)
    return -sequence_logps.float().sum() / float(total)


def gt_anchor_loss(
    logps: torch.Tensor, decision_mask: torch.Tensor
) -> AnchorSetLossOutput:
    """Convenience wrapper for an unchunked GT set."""

    sequence_logps, counts = gt_sequence_logps(logps, decision_mask)
    return AnchorSetLossOutput(
        loss=gt_set_anchor_loss(sequence_logps),
        sequence_logps=sequence_logps,
        decision_counts=counts,
    )


@dataclass(frozen=True)
class AnchorLambdaOutput:
    """The runtime gradient-budget decision for the independent anchor step."""

    lambda_cap: float
    lambda_effective: float
    anchor_to_rl_grad_ratio: float
    update_allowed: bool


def anchor_effective_lambda(
    *,
    max_weight: float,
    rl_reference_grad_norm: float,
    anchor_raw_grad_norm: float,
    target_gradient_ratio: float = 0.10,
    epsilon: float = 1.0e-12,
) -> AnchorLambdaOutput:
    """Cap the anchor's **pre-clip gradient input** at a fraction of RL.

    The returned ratio is ``lambda_effective * raw_anchor_norm / rl_norm``.
    It bounds the gradient handed to clipping/AdamW; it does not claim to bound
    a later AdamW parameter displacement or model-function change.
    """

    values = {
        "max_weight": max_weight,
        "rl_reference_grad_norm": rl_reference_grad_norm,
        "anchor_raw_grad_norm": anchor_raw_grad_norm,
        "target_gradient_ratio": target_gradient_ratio,
        "epsilon": epsilon,
    }
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise ValueError("Anchor gradient-budget values must be finite.")
    if max_weight < 0.0 or target_gradient_ratio < 0.0 or epsilon <= 0.0:
        raise ValueError("Anchor gradient-budget values must be non-negative.")
    if rl_reference_grad_norm <= 0.0 or anchor_raw_grad_norm <= 0.0:
        return AnchorLambdaOutput(0.0, 0.0, 0.0, False)
    lambda_cap = (
        float(target_gradient_ratio) * float(rl_reference_grad_norm)
        / max(float(anchor_raw_grad_norm), float(epsilon))
    )
    effective = min(float(max_weight), lambda_cap)
    ratio = effective * float(anchor_raw_grad_norm) / float(rl_reference_grad_norm)
    return AnchorLambdaOutput(
        lambda_cap=lambda_cap,
        lambda_effective=effective,
        anchor_to_rl_grad_ratio=ratio,
        update_allowed=effective > 0.0,
    )


__all__ = [
    "AnchorLambdaOutput",
    "AnchorSetLossOutput",
    "ClippedChunkOutput",
    "RewardOutput",
    "anchor_effective_lambda",
    "clipped_group_relative_token_sum",
    "group_relative_advantages",
    "group_relative_rewards_and_advantages",
    "gt_all_decision_nll",
    "gt_anchor_loss",
    "gt_sequence_logps",
    "gt_set_anchor_loss",
    "is_effective_rewards",
]
