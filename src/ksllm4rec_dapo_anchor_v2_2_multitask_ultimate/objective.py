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
    """Five-level rewards and per-layer advantages for G=16.

    ``rewards``/``advantages`` are candidate-level scalar reward and its
    group-relative advantage (kept for routing/logging); RL loss uses the
    per-layer ``layer_advantages`` instead.
    """

    rewards: torch.Tensor
    advantages: torch.Tensor
    mean_reward: torch.Tensor
    std_reward: torch.Tensor
    tiers: tuple[str, ...]
    effective: bool
    layer_rewards: torch.Tensor
    layer_advantages: torch.Tensor
    layer_mask: torch.Tensor


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


LAYER_INCREMENT = (
    contract.SID_LAYER_REWARD_DOMAIN,
    contract.SID_LAYER_REWARD_A,
    contract.SID_LAYER_REWARD_B,
    contract.SID_LAYER_REWARD_C,
)

_TIER_TO_T_ERR = {
    "exact": contract.SID_LAYER_COUNT,
    "same_ab": 3,
    "same_a": 2,
    "same_domain": 1,
    "other_domain": 0,
}


def sid_layer_rewards(
    candidate: Sid, positives: Sequence[Sid]
) -> tuple[tuple[float, ...], int]:
    """Return per-layer hit rewards and the first deviation index ``t_err``.

    L1..L4 test domain / a_key / ab_key / exact membership against the GT set.
    SID keys are nested (a_key ⊃ domain, ab_key ⊃ a_key, exact ⊃ ab_key), so
    hitting layer j implies hitting all shallower layers; the first deviation
    cleanly partitions the sequence.  ``t_err`` is the 0-based index of the
    first deviating layer (0..3), or ``SID_LAYER_COUNT`` if all hit.
    """

    domains = {positive.domain for positive in positives}
    a_keys = {positive.a_key for positive in positives}
    ab_keys = {positive.ab_key for positive in positives}
    hits = (
        candidate.domain in domains,
        candidate.a_key in a_keys,
        candidate.ab_key in ab_keys,
        candidate in positives,
    )
    t_err = contract.SID_LAYER_COUNT
    for index, hit in enumerate(hits):
        if not hit:
            t_err = index
            break
    layer = tuple(
        LAYER_INCREMENT[j] if j < t_err else contract.SID_LAYER_REWARD_MISS
        for j in range(contract.SID_LAYER_COUNT)
    )
    return layer, t_err


def group_relative_rewards_and_advantages(
    candidates: Sequence[Sid], positives: Sequence[Sid]
) -> RewardOutput:
    live = tuple(candidates)
    if len(live) != contract.GROUP_SIZE:
        raise ValueError(f"Expected exactly {contract.GROUP_SIZE} candidates.")
    scored = [sid_reward(candidate, positives) for candidate in live]
    rewards = torch.tensor([value for value, _ in scored], dtype=torch.float32)
    layers = [sid_layer_rewards(candidate, positives) for candidate in live]
    layer_rewards = torch.tensor(
        [layer for layer, _ in layers], dtype=torch.float32
    )
    t_errs = [t_err for _, t_err in layers]
    for c, (_, tier) in enumerate(scored):
        expected = _TIER_TO_T_ERR[tier]
        if t_errs[c] != expected:
            raise RuntimeError(
                f"Candidate {c} tier={tier} implies t_err={expected}, "
                f"got t_err={t_errs[c]}."
            )
    layer_mask = torch.tensor(
        [
            [j > t_errs[c] for j in range(contract.SID_LAYER_COUNT)]
            for c in range(contract.GROUP_SIZE)
        ],
        dtype=torch.bool,
    )
    layer_advantages = torch.zeros_like(layer_rewards)
    for j in range(contract.SID_LAYER_COUNT):
        participating = ~layer_mask[:, j]
        if not bool(participating.any()):
            continue
        mean_j = layer_rewards[participating, j].mean()
        layer_advantages[participating, j] = (
            layer_rewards[participating, j] - mean_j
        )
    return RewardOutput(
        rewards=rewards,
        advantages=group_relative_advantages(rewards),
        mean_reward=rewards.mean(),
        std_reward=rewards.std(correction=0),
        tiers=tuple(tier for _, tier in scored),
        effective=is_effective_rewards(rewards),
        layer_rewards=layer_rewards,
        layer_advantages=layer_advantages,
        layer_mask=layer_mask,
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


def _expand_layer_advantages(
    layer_advantages: torch.Tensor, decision_mask: torch.Tensor
) -> torch.Tensor:
    """Spread (C, SID_LAYER_COUNT) per-layer advantages onto (C, T) decision slots.

    The j-th decision token of candidate c (in ``decision_mask`` order) receives
    ``layer_advantages[c, j]``; non-decision slots stay zero.  Raises if a
    candidate does not have exactly ``SID_LAYER_COUNT`` decision tokens.
    """

    count, width = decision_mask.shape
    full = torch.zeros(
        (count, width), dtype=layer_advantages.dtype, device=layer_advantages.device
    )
    for c in range(count):
        positions = decision_mask[c].nonzero(as_tuple=True)[0]
        if int(positions.numel()) != contract.SID_LAYER_COUNT:
            raise ValueError(
                "Every candidate must have exactly SID_LAYER_COUNT decision tokens."
            )
        for j, position in enumerate(positions.tolist()):
            full[c, position] = layer_advantages[c, j]
    return full


def clipped_group_relative_token_sum(
    new_logps: torch.Tensor,
    old_logps: torch.Tensor,
    layer_advantages: torch.Tensor,
    decision_mask: torch.Tensor,
    *,
    clip_low: float = contract.CLIP_RATIO_LOW,
    clip_high: float = contract.CLIP_RATIO_HIGH,
) -> ClippedChunkOutput:
    """Return the asymmetric-clipped RL loss with per-layer advantages.

    Each decision token uses the advantage of its SID layer; masked layers
    (after the first deviation) carry zero advantage and contribute no loss.
    """

    if new_logps.ndim != 2 or not 1 <= new_logps.shape[0] <= contract.LOSS_CHUNK_SIZE:
        raise ValueError("new_logps must have shape (C,T) with 1 <= C <= 8.")
    if old_logps.shape != new_logps.shape:
        raise ValueError("old_logps must match new_logps.")
    if decision_mask.shape != new_logps.shape or decision_mask.dtype != torch.bool:
        raise ValueError("decision_mask must be boolean and match logps.")
    if layer_advantages.shape != (
        new_logps.shape[0],
        contract.SID_LAYER_COUNT,
    ):
        raise ValueError(
            f"layer_advantages must have shape (C, {contract.SID_LAYER_COUNT})."
        )
    if not 0.0 < float(clip_low) < 1.0 < float(clip_high):
        raise ValueError("clip bounds must satisfy 0 < low < 1 < high.")
    if not bool(decision_mask.any()):
        raise ValueError("A clipped chunk needs at least one decision token.")

    new_values = new_logps.float()[decision_mask]
    old_values = old_logps.detach().float()[decision_mask]
    token_advantages = _expand_layer_advantages(
        layer_advantages.detach().float(), decision_mask
    )[decision_mask]
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

    lambda_effective: float
    anchor_to_rl_grad_ratio: float
    update_allowed: bool


def anchor_effective_lambda(
    *,
    rl_reference_grad_norm: float,
    anchor_raw_grad_norm: float,
    target_gradient_ratio: float = 0.30,
    epsilon: float = 1.0e-12,
) -> AnchorLambdaOutput:
    """Normalize the anchor's **pre-clip gradient** to a fraction of RL.

    The anchor gradient is scaled so its pre-clip L2 norm equals
    ``target_gradient_ratio * rl_reference_grad_norm``.  There is no separate
    upper cap: the ratio is the single strength knob.  When either norm is
    non-positive the anchor step is skipped (``update_allowed=False``).
    """

    values = {
        "rl_reference_grad_norm": rl_reference_grad_norm,
        "anchor_raw_grad_norm": anchor_raw_grad_norm,
        "target_gradient_ratio": target_gradient_ratio,
        "epsilon": epsilon,
    }
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise ValueError("Anchor gradient-budget values must be finite.")
    if target_gradient_ratio < 0.0 or epsilon <= 0.0:
        raise ValueError("Anchor gradient-budget values must be non-negative.")
    if rl_reference_grad_norm <= 0.0 or anchor_raw_grad_norm <= 0.0:
        return AnchorLambdaOutput(0.0, 0.0, False)
    lambda_effective = (
        float(target_gradient_ratio) * float(rl_reference_grad_norm)
        / max(float(anchor_raw_grad_norm), float(epsilon))
    )
    ratio = lambda_effective * float(anchor_raw_grad_norm) / float(rl_reference_grad_norm)
    return AnchorLambdaOutput(
        lambda_effective=lambda_effective,
        anchor_to_rl_grad_ratio=ratio,
        update_allowed=lambda_effective > 0.0,
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
