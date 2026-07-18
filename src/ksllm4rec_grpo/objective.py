"""Pure objective functions for the approved GRPO V3.1 experiment."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import torch

from ksllm4rec_orpo.data import Sid


GROUP_SIZE = 8
ADD_GT_PROBABILITY = 0.50
HASH_SEED = 42
ZSCORE_EPSILON = 1.0e-4


@dataclass(frozen=True)
class FinalizedCandidates:
    """Auditable result of the conditional forced-GT decision."""

    live_candidates: tuple[Sid, ...]
    final_candidates: tuple[Sid, ...]
    forced_mask: tuple[bool, ...]
    has_live_gt: bool
    add_gt: bool
    add_gt_uniform: float | None
    forced_gt: Sid | None
    discarded_rollout: Sid | None


@dataclass(frozen=True)
class RewardOutput:
    """Per-group rewards and MiniOneRec-style sample Z-scores."""

    rewards: torch.Tensor
    advantages: torch.Tensor
    mean_reward: torch.Tensor
    std_reward: torch.Tensor
    tiers: tuple[str, ...]


@dataclass(frozen=True)
class GrpoLossOutput:
    """Masked token and candidate losses for one GRPO group."""

    loss: torch.Tensor
    per_candidate_loss: torch.Tensor
    policy_terms: torch.Tensor
    reference_kl: torch.Tensor
    decision_counts: torch.Tensor


def _validate_hash_inputs(group_id: str, epoch_index: int, seed: int) -> None:
    if not isinstance(group_id, str) or not group_id:
        raise ValueError("group_id must be a non-empty string.")
    if epoch_index <= 0:
        raise ValueError("epoch_index must be positive.")
    if seed < 0:
        raise ValueError("seed must be non-negative.")


def deterministic_uniform(
    namespace: str,
    *,
    group_id: str,
    epoch_index: int,
    seed: int = HASH_SEED,
) -> float:
    """Map a namespaced group/epoch key to a reproducible value in [0, 1)."""

    _validate_hash_inputs(group_id, epoch_index, seed)
    if not namespace or "|" in namespace:
        raise ValueError("namespace must be non-empty and cannot contain '|'.")
    payload = f"{namespace}|{seed}|{epoch_index}|{group_id}".encode()
    first_eight_bytes = hashlib.sha256(payload).digest()[:8]
    return int.from_bytes(first_eight_bytes, "big", signed=False) / 2**64


def should_add_gt(
    *,
    group_id: str,
    epoch_index: int,
    probability: float = ADD_GT_PROBABILITY,
    seed: int = HASH_SEED,
) -> tuple[bool, float]:
    """Return the deterministic add-GT decision and its audit value."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1].")
    uniform = deterministic_uniform(
        "add_gt", group_id=group_id, epoch_index=epoch_index, seed=seed
    )
    return uniform < probability, uniform


def select_forced_gt(
    positives: Sequence[Sid],
    *,
    group_id: str,
    epoch_index: int,
    seed: int = HASH_SEED,
) -> Sid:
    """Select one of a group's GT SIDs with the frozen SHA256 rule."""

    _validate_hash_inputs(group_id, epoch_index, seed)
    ordered = tuple(sorted(positives, key=Sid.render))
    if not ordered:
        raise ValueError("Every recommendation group must have at least one GT SID.")
    if len(set(ordered)) != len(ordered):
        raise ValueError("positive SIDs must be unique within a group.")
    payload = f"gt_index|{seed}|{epoch_index}|{group_id}".encode()
    number = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)
    return ordered[number % len(ordered)]


def finalize_candidates(
    live_candidates: Sequence[Sid],
    positives: Sequence[Sid],
    *,
    group_id: str,
    epoch_index: int,
    probability: float = ADD_GT_PROBABILITY,
    seed: int = HASH_SEED,
) -> FinalizedCandidates:
    """Conditionally replace live candidate index 7 with a deterministic GT."""

    live = tuple(live_candidates)
    positive_tuple = tuple(positives)
    if len(live) != GROUP_SIZE:
        raise ValueError(f"Expected exactly {GROUP_SIZE} live candidates.")
    if not positive_tuple:
        raise ValueError("Every recommendation group must have at least one GT SID.")

    positive_set = set(positive_tuple)
    if len(positive_set) != len(positive_tuple):
        raise ValueError("positive SIDs must be unique within a group.")
    has_live_gt = any(candidate in positive_set for candidate in live)
    if has_live_gt:
        return FinalizedCandidates(
            live_candidates=live,
            final_candidates=live,
            forced_mask=(False,) * GROUP_SIZE,
            has_live_gt=True,
            add_gt=False,
            add_gt_uniform=None,
            forced_gt=None,
            discarded_rollout=None,
        )

    add_gt, uniform = should_add_gt(
        group_id=group_id,
        epoch_index=epoch_index,
        probability=probability,
        seed=seed,
    )
    if not add_gt:
        return FinalizedCandidates(
            live_candidates=live,
            final_candidates=live,
            forced_mask=(False,) * GROUP_SIZE,
            has_live_gt=False,
            add_gt=False,
            add_gt_uniform=uniform,
            forced_gt=None,
            discarded_rollout=None,
        )

    forced_gt = select_forced_gt(
        positive_tuple,
        group_id=group_id,
        epoch_index=epoch_index,
        seed=seed,
    )
    final = live[: GROUP_SIZE - 1] + (forced_gt,)
    return FinalizedCandidates(
        live_candidates=live,
        final_candidates=final,
        forced_mask=(False,) * (GROUP_SIZE - 1) + (True,),
        has_live_gt=False,
        add_gt=True,
        add_gt_uniform=uniform,
        forced_gt=forced_gt,
        discarded_rollout=live[-1],
    )


def sid_reward(candidate: Sid, positives: Sequence[Sid]) -> tuple[float, str]:
    """Return the highest hierarchical reward against every valid GT."""

    if not positives:
        raise ValueError("Every recommendation group must have at least one GT SID.")
    if candidate in positives:
        return 1.0, "exact"
    if any(candidate.ab_key == positive.ab_key for positive in positives):
        return 0.20, "same_ab"
    if any(candidate.a_key == positive.a_key for positive in positives):
        return 0.05, "same_a"
    if any(candidate.domain == positive.domain for positive in positives):
        return 0.01, "same_domain"
    return 0.0, "other_domain"


def rewards_and_advantages(
    candidates: Sequence[Sid],
    positives: Sequence[Sid],
    *,
    epsilon: float = ZSCORE_EPSILON,
) -> RewardOutput:
    """Score one group and compute FP32 sample-standard-deviation Z-scores."""

    if len(candidates) < 2:
        raise ValueError("Z-score requires at least two candidates.")
    if epsilon <= 0.0:
        raise ValueError("epsilon must be positive.")
    scored = [sid_reward(candidate, positives) for candidate in candidates]
    rewards = torch.tensor([value for value, _ in scored], dtype=torch.float32)
    mean_reward = rewards.mean()
    if bool(torch.equal(rewards, rewards[0].expand_as(rewards))):
        std_reward = torch.zeros((), dtype=torch.float32)
        advantages = torch.zeros_like(rewards)
    else:
        std_reward = rewards.std(correction=1)
        advantages = (rewards - mean_reward) / (std_reward + float(epsilon))
    if not bool(torch.isfinite(advantages).all()):
        raise FloatingPointError("Reward advantages contain NaN or Inf.")
    return RewardOutput(
        rewards=rewards,
        advantages=advantages,
        mean_reward=mean_reward,
        std_reward=std_reward,
        tiers=tuple(tier for _, tier in scored),
    )


def grpo_loss(
    new_logps: torch.Tensor,
    reference_logps: torch.Tensor,
    old_logps: torch.Tensor,
    advantages: torch.Tensor,
    decision_mask: torch.Tensor,
    forced_mask: torch.Tensor,
    *,
    clip_epsilon: float = 0.20,
    beta: float = 0.02,
) -> GrpoLossOutput:
    """Compute the V3.1 loss with ragged decision-token masking.

    ``new_logps`` and ``reference_logps`` have shape ``(G, T)``.  The boolean
    ``decision_mask`` has the same shape and excludes padding and deterministic
    one-child trie steps. ``forced_mask`` has shape ``(G,)``. ``old_logps`` has
    shape ``(O, T)``, where ``O == (~forced_mask).sum()``; its rows correspond
    in order to non-forced candidates, so a forced GT has no old-policy row.
    """

    if new_logps.ndim != 2:
        raise ValueError("new_logps must have shape (G, T).")
    if reference_logps.shape != new_logps.shape:
        raise ValueError("reference_logps must match new_logps shape (G, T).")
    if reference_logps.device != new_logps.device:
        raise ValueError("reference_logps and new_logps must share one device.")
    group_size, token_width = new_logps.shape
    if advantages.shape != (group_size,):
        raise ValueError("advantages must have shape (G,).")
    if decision_mask.shape != new_logps.shape or decision_mask.dtype != torch.bool:
        raise ValueError("decision_mask must be boolean with shape (G, T).")
    if forced_mask.shape != (group_size,) or forced_mask.dtype != torch.bool:
        raise ValueError("forced_mask must be boolean with shape (G,).")
    on_policy_rows = ~forced_mask
    on_policy_count = int(on_policy_rows.sum().item())
    if old_logps.shape != (on_policy_count, token_width):
        raise ValueError("old_logps must have shape ((~forced_mask).sum(), T).")
    if old_logps.device != new_logps.device:
        raise ValueError("old_logps and new_logps must share one device.")
    if not 0.0 < clip_epsilon < 1.0:
        raise ValueError("clip_epsilon must be in (0, 1).")
    if beta < 0.0:
        raise ValueError("beta must be non-negative.")

    decision_mask = decision_mask.to(new_logps.device)
    forced_mask = forced_mask.to(new_logps.device)
    counts = decision_mask.sum(dim=-1)
    if bool(counts.eq(0).any()):
        raise ValueError("Every candidate must have at least one decision token.")

    new_fp32 = new_logps.float()
    reference_fp32 = reference_logps.detach().float()
    advantages_fp32 = advantages.detach().to(new_logps.device, torch.float32)
    on_policy_rows = ~forced_mask

    new_values = new_fp32[decision_mask]
    reference_values = reference_fp32[decision_mask]
    if not bool(torch.isfinite(new_values).all()):
        raise FloatingPointError("new_logps contain NaN or Inf at decision tokens.")
    if not bool(torch.isfinite(reference_values).all()):
        raise FloatingPointError(
            "reference_logps contain NaN or Inf at decision tokens."
        )
    if not bool(torch.isfinite(advantages_fp32).all()):
        raise FloatingPointError("advantages contain NaN or Inf.")

    token_policy_terms = torch.zeros_like(new_fp32)
    live_decision_mask = decision_mask[on_policy_rows]
    if on_policy_count:
        old_values = old_logps.detach().float()[live_decision_mask]
        live_new_values = new_fp32[on_policy_rows][live_decision_mask]
        if not bool(torch.isfinite(old_values).all()):
            raise FloatingPointError("old_logps contain NaN or Inf at decision tokens.")
        ratios = torch.exp(live_new_values - old_values)
        live_advantages = advantages_fp32[on_policy_rows]
        live_token_advantages = live_advantages[:, None].expand_as(live_decision_mask)[
            live_decision_mask
        ]
        clipped_ratios = ratios.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
        live_terms = torch.minimum(
            ratios * live_token_advantages,
            clipped_ratios * live_token_advantages,
        )
        live_term_rows = torch.zeros_like(new_fp32[on_policy_rows])
        live_term_rows[live_decision_mask] = live_terms
        token_policy_terms[on_policy_rows] = live_term_rows

    if bool(forced_mask.any()):
        forced_decision_mask = decision_mask[forced_mask]
        forced_new_values = new_fp32[forced_mask][forced_decision_mask]
        forced_advantages = advantages_fp32[forced_mask]
        forced_token_advantages = forced_advantages[:, None].expand_as(
            forced_decision_mask
        )[forced_decision_mask]
        forced_terms = (
            torch.exp(forced_new_values - forced_new_values.detach())
            * forced_token_advantages
        )
        forced_term_rows = torch.zeros_like(new_fp32[forced_mask])
        forced_term_rows[forced_decision_mask] = forced_terms
        token_policy_terms[forced_mask] = forced_term_rows

    d = reference_values - new_values
    kl_values = torch.exp(d) - d - 1.0
    token_kl = torch.zeros_like(new_fp32)
    token_kl[decision_mask] = kl_values
    token_losses = -(token_policy_terms - float(beta) * token_kl)
    per_candidate_loss = token_losses.sum(dim=-1) / counts.to(torch.float32)
    loss = per_candidate_loss.mean()
    if not bool(torch.isfinite(loss)):
        raise FloatingPointError("GRPO loss is NaN or Inf.")
    return GrpoLossOutput(
        loss=loss,
        per_candidate_loss=per_candidate_loss,
        policy_terms=token_policy_terms,
        reference_kl=token_kl,
        decision_counts=counts,
    )
