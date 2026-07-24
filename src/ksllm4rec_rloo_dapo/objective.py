"""Reward, effective-group filtering, and asymmetric clipped RLOO loss."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from ksllm4rec_orpo.data import Sid
from ksllm4rec_rloo.objective import sid_reward

from . import contract


@dataclass(frozen=True)
class RewardOutput:
    rewards: torch.Tensor
    advantages: torch.Tensor
    mean_reward: torch.Tensor
    tiers: tuple[str, ...]
    effective: bool


@dataclass(frozen=True)
class ClippedChunkOutput:
    loss_sum: torch.Tensor
    per_candidate_loss_sums: torch.Tensor
    ratios: torch.Tensor
    clip_active: torch.Tensor
    below_low: torch.Tensor
    above_high: torch.Tensor
    decision_counts: torch.Tensor


def is_effective_rewards(rewards: torch.Tensor) -> bool:
    if rewards.shape != (contract.GROUP_SIZE,):
        raise ValueError(f"rewards must have shape ({contract.GROUP_SIZE},).")
    values = rewards.detach().float()
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("rewards contain NaN or Inf.")
    return bool(values.max().item() > values.min().item())


def rloo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    if rewards.shape != (contract.GROUP_SIZE,):
        raise ValueError(f"rewards must have shape ({contract.GROUP_SIZE},).")
    values = rewards.float()
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("rewards contain NaN or Inf.")
    return (contract.GROUP_SIZE / (contract.GROUP_SIZE - 1)) * (
        values - values.mean()
    )


def rewards_and_advantages(
    candidates: Sequence[Sid], positives: Sequence[Sid]
) -> RewardOutput:
    live = tuple(candidates)
    if len(live) != contract.GROUP_SIZE:
        raise ValueError(f"Expected exactly {contract.GROUP_SIZE} candidates.")
    scored = [sid_reward(candidate, positives) for candidate in live]
    rewards = torch.tensor([value for value, _ in scored], dtype=torch.float32)
    return RewardOutput(
        rewards=rewards,
        advantages=rloo_advantages(rewards),
        mean_reward=rewards.mean(),
        tiers=tuple(tier for _, tier in scored),
        effective=is_effective_rewards(rewards),
    )


def clipped_rloo_token_sum(
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    decision_mask: torch.Tensor,
    *,
    clip_low: float = contract.CLIP_RATIO_LOW,
    clip_high: float = contract.CLIP_RATIO_HIGH,
) -> ClippedChunkOutput:
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
        raise ValueError("A clipped RLOO chunk needs at least one decision token.")

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


__all__ = [
    "ClippedChunkOutput",
    "RewardOutput",
    "clipped_rloo_token_sum",
    "is_effective_rewards",
    "rewards_and_advantages",
    "rloo_advantages",
]
