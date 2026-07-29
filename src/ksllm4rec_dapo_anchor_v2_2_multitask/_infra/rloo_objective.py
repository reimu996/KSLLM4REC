"""Pure reward, RLOO, and conditional GT-set-anchor objectives."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable, Sequence

import torch

from .orpo_data import Sid

from .rloo_contract import (
    ANCHOR_MAX_WEIGHT,
    ANCHOR_TARGET_GRADIENT_RATIO,
    GROUP_SIZE,
    MIN_GRAD_NORM,
    REWARD_VALUES,
)


class ObjectiveBranch(StrEnum):
    RLOO = "rloo"
    GT_SET_ANCHOR = "gt_set_anchor"
    SKIP = "skip"


@dataclass(frozen=True)
class RewardOutput:
    """Fixed rewards and leave-one-out advantages for one G=16 rollout."""

    rewards: torch.Tensor
    advantages: torch.Tensor
    mean_reward: torch.Tensor
    tiers: tuple[str, ...]
    branch: ObjectiveBranch


@dataclass(frozen=True)
class RlooLossOutput:
    """Decision-token-normalized policy loss for one signal-bearing group."""

    loss: torch.Tensor
    per_candidate_loss: torch.Tensor
    policy_terms: torch.Tensor
    decision_counts: torch.Tensor


@dataclass(frozen=True)
class RlooChunkLossOutput:
    """One memory-bounded candidate chunk using the full-group denominator."""

    loss: torch.Tensor
    per_candidate_loss: torch.Tensor
    decision_counts: torch.Tensor


@dataclass(frozen=True)
class AnchorLossOutput:
    """Probability-mass loss over every unique GT SID in one group."""

    loss: torch.Tensor
    sequence_logps: torch.Tensor
    decision_counts: torch.Tensor


@dataclass(frozen=True)
class WindowLossOutput:
    """Separately averaged RLOO and anchor branches for one source window."""

    loss: torch.Tensor | None
    rloo_mean: torch.Tensor | None
    anchor_mean: torch.Tensor | None
    weighted_anchor: torch.Tensor | None
    rloo_groups: int
    anchor_groups: int
    skip_groups: int


def _validate_unique_positives(positives: Sequence[Sid]) -> tuple[Sid, ...]:
    values = tuple(positives)
    if not values:
        raise ValueError("Every recommendation group must have at least one GT SID.")
    if not all(isinstance(value, Sid) for value in values):
        raise TypeError("Every GT must be a Sid.")
    if len(set(values)) != len(values):
        raise ValueError("GT SIDs must be unique within a group.")
    return values


def sid_reward(candidate: Sid, positives: Sequence[Sid]) -> tuple[float, str]:
    """Return the highest of the five frozen hierarchy rewards."""

    if not isinstance(candidate, Sid):
        raise TypeError("candidate must be a Sid.")
    gt = _validate_unique_positives(positives)
    if candidate in gt:
        return REWARD_VALUES["exact"], "exact"
    if any(candidate.ab_key == positive.ab_key for positive in gt):
        return REWARD_VALUES["same_ab"], "same_ab"
    if any(candidate.a_key == positive.a_key for positive in gt):
        return REWARD_VALUES["same_a"], "same_a"
    if any(candidate.domain == positive.domain for positive in gt):
        return REWARD_VALUES["same_domain"], "same_domain"
    return REWARD_VALUES["other_domain"], "other_domain"


def rloo_advantages(rewards: torch.Tensor) -> torch.Tensor:
    """Compute A_i = G/(G-1) * (r_i - mean(r)) without std scaling."""

    if rewards.ndim != 1 or rewards.shape[0] != GROUP_SIZE:
        raise ValueError(f"rewards must have shape ({GROUP_SIZE},).")
    values = rewards.to(dtype=torch.float32)
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("rewards contain NaN or Inf.")
    advantages = (GROUP_SIZE / (GROUP_SIZE - 1)) * (values - values.mean())
    if bool(torch.equal(values, values[0].expand_as(values))):
        advantages = torch.zeros_like(values)
    if not bool(torch.isfinite(advantages).all()):
        raise FloatingPointError("RLOO advantages contain NaN or Inf.")
    return advantages


def select_objective_branch(rewards: torch.Tensor) -> ObjectiveBranch:
    """Choose exactly one of RLOO, anchor, or already-solved skip."""

    if rewards.ndim != 1 or rewards.shape[0] != GROUP_SIZE:
        raise ValueError(f"rewards must have shape ({GROUP_SIZE},).")
    values = rewards.detach().to(dtype=torch.float32)
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("rewards contain NaN or Inf.")
    if not bool(torch.equal(values, values[0].expand_as(values))):
        return ObjectiveBranch.RLOO
    if bool(values.eq(REWARD_VALUES["exact"]).all()):
        return ObjectiveBranch.SKIP
    return ObjectiveBranch.GT_SET_ANCHOR


def rewards_and_advantages(
    candidates: Sequence[Sid],
    positives: Sequence[Sid],
) -> RewardOutput:
    """Score exactly 16 live candidates; no candidate is replaced or injected."""

    live = tuple(candidates)
    if len(live) != GROUP_SIZE:
        raise ValueError(f"Expected exactly {GROUP_SIZE} live candidates.")
    gt = _validate_unique_positives(positives)
    scored = [sid_reward(candidate, gt) for candidate in live]
    rewards = torch.tensor([reward for reward, _ in scored], dtype=torch.float32)
    return RewardOutput(
        rewards=rewards,
        advantages=rloo_advantages(rewards),
        mean_reward=rewards.mean(),
        tiers=tuple(tier for _, tier in scored),
        branch=select_objective_branch(rewards),
    )


score_candidates = rewards_and_advantages


def rloo_loss(
    logps: torch.Tensor,
    advantages: torch.Tensor,
    decision_mask: torch.Tensor,
) -> RlooLossOutput:
    """Apply detached RLOO advantages only at non-deterministic trie steps.

    ``logps`` and ``decision_mask`` have shape ``(16, T)``. Padding and
    one-child trie transitions must be false in ``decision_mask``. The group
    loss denominator is the total number of true decision tokens across all
    candidates, exactly matching Spec V2.0.
    """

    if logps.ndim != 2 or logps.shape[0] != GROUP_SIZE:
        raise ValueError(f"logps must have shape ({GROUP_SIZE}, T).")
    if decision_mask.shape != logps.shape or decision_mask.dtype != torch.bool:
        raise ValueError("decision_mask must be boolean with shape (16, T).")
    if advantages.shape != (GROUP_SIZE,):
        raise ValueError(f"advantages must have shape ({GROUP_SIZE},).")
    if decision_mask.device != logps.device or advantages.device != logps.device:
        raise ValueError("logps, advantages, and decision_mask must share one device.")

    counts = decision_mask.sum(dim=-1)
    total_decisions = int(counts.sum().item())
    if total_decisions <= 0:
        raise ValueError("An RLOO group must contain at least one decision token.")
    values = logps.float()[decision_mask]
    detached_advantages = advantages.detach().float()
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("logps contain NaN or Inf at decision tokens.")
    if not bool(torch.isfinite(detached_advantages).all()):
        raise FloatingPointError("advantages contain NaN or Inf.")
    if abs(float(detached_advantages.sum().item())) > 1.0e-5:
        raise ValueError("RLOO advantages must sum to zero within 1e-5.")

    expanded_advantages = detached_advantages[:, None].expand_as(logps)
    terms = values * expanded_advantages[decision_mask]
    token_terms = torch.zeros_like(logps, dtype=torch.float32)
    token_terms[decision_mask] = terms
    denominator = torch.tensor(
        total_decisions, dtype=torch.float32, device=logps.device
    )
    per_candidate_loss = -token_terms.sum(dim=-1) / denominator
    loss = per_candidate_loss.sum()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("RLOO loss is NaN or Inf.")
    return RlooLossOutput(
        loss=loss,
        per_candidate_loss=per_candidate_loss,
        policy_terms=token_terms,
        decision_counts=counts,
    )


def rloo_chunk_loss(
    logps: torch.Tensor,
    advantages: torch.Tensor,
    decision_mask: torch.Tensor,
    *,
    total_group_decisions: int,
) -> RlooChunkLossOutput:
    """Compute one C<=8 chunk while preserving the full-group denominator.

    Summing the two G=16 chunk losses is exactly equal to :func:`rloo_loss`.
    This lets the trainer backpropagate each long-context graph immediately.
    """

    if logps.ndim != 2 or not 1 <= logps.shape[0] <= 8:
        raise ValueError("chunk logps must have shape (C, T) with 1 <= C <= 8.")
    if decision_mask.shape != logps.shape or decision_mask.dtype != torch.bool:
        raise ValueError("chunk decision_mask must be boolean with shape (C, T).")
    if advantages.shape != (logps.shape[0],):
        raise ValueError("chunk advantages must have shape (C,).")
    if decision_mask.device != logps.device or advantages.device != logps.device:
        raise ValueError("chunk tensors must share one device.")
    if (
        isinstance(total_group_decisions, bool)
        or not isinstance(total_group_decisions, int)
        or total_group_decisions <= 0
    ):
        raise ValueError("total_group_decisions must be a positive integer.")
    counts = decision_mask.sum(dim=-1)
    values = logps.float()[decision_mask]
    detached_advantages = advantages.detach().float()
    if not bool(torch.isfinite(values).all()) or not bool(
        torch.isfinite(detached_advantages).all()
    ):
        raise FloatingPointError("chunk RLOO inputs contain NaN or Inf.")
    expanded = detached_advantages[:, None].expand_as(logps)
    terms = values * expanded[decision_mask]
    token_terms = torch.zeros_like(logps, dtype=torch.float32)
    token_terms[decision_mask] = terms
    denominator = torch.tensor(
        total_group_decisions, dtype=torch.float32, device=logps.device
    )
    per_candidate = -token_terms.sum(dim=-1) / denominator
    loss = per_candidate.sum()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("chunk RLOO loss is NaN or Inf.")
    return RlooChunkLossOutput(
        loss=loss,
        per_candidate_loss=per_candidate,
        decision_counts=counts,
    )


def gt_sequence_logps(
    logps: torch.Tensor,
    decision_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reduce token log-probabilities to one score per GT before chunk merging."""

    if logps.ndim != 2 or logps.shape[0] < 1:
        raise ValueError("GT logps must have shape (M, T) with M >= 1.")
    if decision_mask.shape != logps.shape or decision_mask.dtype != torch.bool:
        raise ValueError("GT decision_mask must be boolean with shape (M, T).")
    if decision_mask.device != logps.device:
        raise ValueError("GT logps and decision_mask must share one device.")
    values = logps.float()[decision_mask]
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("GT logps contain NaN or Inf at decision tokens.")
    token_values = torch.zeros_like(logps, dtype=torch.float32)
    token_values[decision_mask] = values
    return token_values.sum(dim=-1), decision_mask.sum(dim=-1)


def gt_set_anchor_loss(sequence_logps: torch.Tensor) -> torch.Tensor:
    """Compute one global ``-logsumexp`` after all GT chunks are concatenated."""

    if sequence_logps.ndim != 1 or sequence_logps.numel() < 1:
        raise ValueError("sequence_logps must have shape (M,) with M >= 1.")
    values = sequence_logps.float()
    if not bool(torch.isfinite(values).all()):
        raise FloatingPointError("GT sequence_logps contain NaN or Inf.")
    loss = -torch.logsumexp(values, dim=0)
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("GT-set anchor loss is NaN or Inf.")
    return loss


def gt_anchor_loss(
    logps: torch.Tensor,
    decision_mask: torch.Tensor,
) -> AnchorLossOutput:
    """Convenience wrapper for an unchunked GT set."""

    sequence_logps, counts = gt_sequence_logps(logps, decision_mask)
    return AnchorLossOutput(
        loss=gt_set_anchor_loss(sequence_logps),
        sequence_logps=sequence_logps,
        decision_counts=counts,
    )


def _stack_scalar_losses(
    values: Iterable[torch.Tensor], label: str
) -> tuple[torch.Tensor, ...]:
    losses = tuple(values)
    for loss in losses:
        if not isinstance(loss, torch.Tensor) or loss.ndim != 0:
            raise ValueError(f"Every {label} loss must be a scalar tensor.")
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError(f"{label} loss contains NaN or Inf.")
    return losses


def window_loss(
    rloo_losses: Iterable[torch.Tensor],
    anchor_losses: Iterable[torch.Tensor],
    *,
    anchor_weight: float,
    skip_groups: int = 0,
) -> WindowLossOutput:
    """Average each branch independently, then apply the calibrated anchor weight."""

    if not math.isfinite(anchor_weight) or not 0.0 <= anchor_weight <= ANCHOR_MAX_WEIGHT:
        raise ValueError(
            f"anchor_weight must be finite and in [0, {ANCHOR_MAX_WEIGHT}]."
        )
    if not isinstance(skip_groups, int) or skip_groups < 0:
        raise ValueError("skip_groups must be a non-negative integer.")
    rloo = _stack_scalar_losses(rloo_losses, "RLOO")
    anchors = _stack_scalar_losses(anchor_losses, "anchor")
    rloo_mean = torch.stack(rloo).mean() if rloo else None
    anchor_mean = torch.stack(anchors).mean() if anchors else None
    weighted_anchor = (
        anchor_mean * float(anchor_weight) if anchor_mean is not None else None
    )
    if rloo_mean is not None and weighted_anchor is not None:
        if rloo_mean.device != weighted_anchor.device:
            raise ValueError("RLOO and anchor losses must share one device.")
        total = rloo_mean + weighted_anchor
    elif rloo_mean is not None:
        total = rloo_mean
    else:
        total = weighted_anchor
    return WindowLossOutput(
        loss=total,
        rloo_mean=rloo_mean,
        anchor_mean=anchor_mean,
        weighted_anchor=weighted_anchor,
        rloo_groups=len(rloo),
        anchor_groups=len(anchors),
        skip_groups=skip_groups,
    )


def calibrate_anchor_lambda(
    rloo_grad_norm: float,
    anchor_grad_norm: float,
    *,
    target_ratio: float = ANCHOR_TARGET_GRADIENT_RATIO,
    max_weight: float = ANCHOR_MAX_WEIGHT,
    min_grad_norm: float = MIN_GRAD_NORM,
) -> float:
    """Pure Spec V2.0 calibration: min(max_weight, ratio * ||g_rl||/||g_h||)."""

    values = (rloo_grad_norm, anchor_grad_norm, target_ratio, max_weight, min_grad_norm)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("Anchor calibration inputs must be finite.")
    if rloo_grad_norm <= min_grad_norm or anchor_grad_norm <= min_grad_norm:
        raise ValueError("Both calibration gradient norms must exceed min_grad_norm.")
    if target_ratio <= 0.0 or max_weight <= 0.0 or min_grad_norm <= 0.0:
        raise ValueError("Calibration ratios, cap, and minimum norm must be positive.")
    return min(
        float(max_weight),
        float(target_ratio) * float(rloo_grad_norm) / float(anchor_grad_norm),
    )
