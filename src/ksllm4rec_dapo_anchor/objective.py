"""Reward, effective-group filtering, group-relative advantage, GT anchor loss.

Difference from ksllm4rec_rloo_dapo.objective:
  - advantage 改用 DAPO/GRPO 公式:  A_i = (r_i - mean) / std
    (不再用 leave-one-out scaling factor G/(G-1))
  - 新增 gt_set_anchor_loss: 给一组 prompt->GT teacher-forcing 拉高 GT 概率.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch.nn.functional import log_softmax

from ._infra.orpo_data import Sid
from ._infra.rloo_objective import sid_reward

from . import contract


# ── Rewards / advantages ─────────────────────────────────────────────

@dataclass(frozen=True)
class RewardOutput:
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
    """DAPO/GRPO advantage: (r - mean) / std.

    std 加 eps 防 0 (effective 组不会全相同, 但等价防线).
    """
    if rewards.shape != (contract.GROUP_SIZE,):
        raise ValueError(f"rewards must have shape ({contract.GROUP_SIZE},).")
    values = rewards.float()
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("rewards contain NaN or Inf.")
    std = values.std(correction=0)
    if not bool(torch.isfinite(std).all()) or float(std.item()) <= 0.0:
        raise FloatingPointError("Group std is non-positive; not effective.")
    return (values - values.mean()) / (std + 1.0e-6)


def group_relative_rewards_and_advantages(
    candidates: Sequence[Sid], positives: Sequence[Sid]
) -> RewardOutput:
    live = tuple(candidates)
    if len(live) != contract.GROUP_SIZE:
        raise ValueError(f"Expected exactly {contract.GROUP_SIZE} candidates.")
    scored = [sid_reward(candidate, positives) for candidate in live]
    rewards = torch.tensor(
        [value for value, _ in scored], dtype=torch.float32
    )
    advantages = group_relative_advantages(rewards)
    return RewardOutput(
        rewards=rewards,
        advantages=advantages,
        mean_reward=rewards.mean(),
        std_reward=rewards.std(correction=0),
        tiers=tuple(tier for _, tier in scored),
        effective=is_effective_rewards(rewards),
    )


# ── Clipped loss ──────────────────────────────────────────────────────

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


# ── GT anchor loss ────────────────────────────────────────────────────

@dataclass(frozen=True)
class AnchorLossOutput:
    """GT teacher-forced anchor loss for one prompt->GT pair.

    L = -logsumexp(logits[GT_indices]) / num_decision_tokens
      ? Wait — should be the SUM of GT decision token logps normalized.
        For each decision position t in GT completion:
          loss_t = -logp(GT_t | prompt, GT_<t)
        anchor_loss = mean_t (loss_t)
        (NOT logsumexp across tokens — we want every GT token to be pushed up.)
    """

    anchor_loss: torch.Tensor
    decision_token_count: int
    per_token_logps: torch.Tensor
    gt_token_ids: torch.Tensor


def gt_set_anchor_loss(
    new_logps: torch.Tensor,
    decision_mask: torch.Tensor,
    *,
    completion_width: int = contract.MAX_COMPLETION_LENGTH,
) -> AnchorLossOutput:
    """Compute the GT teacher-forcing anchor loss.

    new_logps:      (1, T) — logp of GT token at each position under current policy.
                              (computed from teacher-forced forward of GT sequence)
    decision_mask: (1, T) — which positions are decision tokens (grammar branching
                              points; non-decision positions don't contribute)

    Returns negative mean of GT decision logps:
        anchor_loss = -sum(new_logps[mask]) / sum(mask)

    This is the same NLL style as SFT but only over decision positions; non-decision
    tokens (forced by grammar) naturally carry logp=0.
    """
    if new_logps.ndim != 2 or new_logps.shape[0] != 1:
        raise ValueError("new_logps must have shape (1,T).")
    if decision_mask.shape != new_logps.shape or decision_mask.dtype != torch.bool:
        raise ValueError("decision_mask must be boolean and match new_logps.")
    if not bool(decision_mask.any()):
        raise ValueError("Anchor loss requires at least one GT decision token.")

    decision_logps = new_logps.float()[decision_mask]
    if not bool(torch.isfinite(decision_logps).all()):
        raise FloatingPointError("GT anchor logps contain NaN or Inf.")

    decision_count = int(decision_mask.sum().item())
    loss = -decision_logps.sum() / float(decision_count)
    return AnchorLossOutput(
        anchor_loss=loss,
        decision_token_count=decision_count,
        per_token_logps=decision_logps.detach(),
        gt_token_ids=torch.zeros((), dtype=torch.long),  # placeholder for API symmetry
    )


__all__ = [
    "AnchorLossOutput",
    "ClippedChunkOutput",
    "RewardOutput",
    "clipped_group_relative_token_sum",
    "group_relative_advantages",
    "group_relative_rewards_and_advantages",
    "gt_set_anchor_loss",
    "is_effective_rewards",
]
