"""DAPO-Anchor V2.2: four on-policy RL steps plus one bounded GT-set step.

From W0, every reward-flat/no-exact group contributes to one window-level
GT-set gradient.  The group count is never sampled or capped.  The raw anchor
gradient is scaled so its pre-clip norm is at most 10% of the median of the four
RL pre-clip norms, then at most one independent optimizer step is taken.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import random
from statistics import median
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ._infra.grpo_constraint import RecommendationGrammar
from ._infra.grpo_prompt import encode_prompt
from ._infra.grpo_trie import SidPrefixTrie
from ._infra.orpo_data import Sid
from ._infra.rloo_checkpoint import save_policy_atomic
from ._infra.rloo_modeling import PolicyModel, load_policy_model
from ._infra.dapo_checkpoint import (
    RecoveryCursor,
    load_latest_recovery,
    recovery_checkpoint_due,
    restore_training_state,
    save_recovery_checkpoint,
)
from ._infra.dapo_fingerprint import validate_runtime_signature
from ._infra.dapo_rollout import (
    CacheRolloutStats,
    CanonicalCandidate,
    PromptRequest,
    canonicalize_prompt_rollout,
    rollout_prompt_batch,
)
from ._infra.dapo_sampling import (
    DeterministicGroupStream,
    SourceCursor,
    append_effective_groups,
)
from ._infra.dapo_scoring import (
    dense_scoring_mode,
    grammar_completion_width,
    score_completions_dense_fixed,
)
from .anchor_scoring import score_prompt_completion_pairs_dense
from . import contract
from .data import (
    SidTargetGroup,
    SidTask,
    load_recommendation_target_groups,
    load_text_to_sid_target_groups,
)
from .objective import (
    AnchorLambdaOutput,
    RewardOutput,
    anchor_effective_lambda,
    clipped_group_relative_token_sum,
    group_relative_rewards_and_advantages,
    gt_sequence_logps,
    gt_set_anchor_loss,
)


# ── Data structures ────────────────────────────────────────────────────

@dataclass(frozen=True)
class PreparedEffectiveGroup:
    """One effective prompt->candidates group for RL phase."""

    group: SidTargetGroup
    prompt_ids: tuple[int, ...]
    positives: tuple[Sid, ...]
    candidates: tuple[CanonicalCandidate, ...]
    reward: RewardOutput
    proposal_target_max_logp_difference: float
    corrected_candidate_count: int
    target_forward_calls: int


@dataclass(frozen=True)
class PreparedFilterGroup:
    """One reward-flat/no-exact group eligible for the GT-set anchor."""

    group: SidTargetGroup
    prompt_ids: tuple[int, ...]
    positives: tuple[Sid, ...]


@dataclass(frozen=True)
class RawBatchAudit:
    raw_group_ids: tuple[str, ...]
    effective_group_ids: tuple[str, ...]
    filtered_group_ids: tuple[str, ...]
    anchor_candidate_group_ids: tuple[str, ...]
    overflow_group_ids: tuple[str, ...]
    cache_stats: CacheRolloutStats


@dataclass(frozen=True)
class WindowCollection:
    groups: tuple[PreparedEffectiveGroup, ...]
    anchor_candidates: tuple[PreparedFilterGroup, ...]
    source_cursor: SourceCursor
    raw_prompt_count: int
    filtered_group_count: int
    overflow_group_count: int
    audits: tuple[RawBatchAudit, ...]


@dataclass(frozen=True)
class OptimizerStepResult:
    loss: float
    learning_rate: float
    decision_tokens: int
    gradient_norm: float
    max_replay_logp_difference: float
    ratio_min: float
    ratio_max: float
    ratio_mean: float
    clipped_tokens: int
    below_low_tokens: int
    above_high_tokens: int
    group_ids: tuple[str, ...]


@dataclass(frozen=True)
class AnchorStepResult:
    candidate_count: int
    groups_used: int
    group_ids: tuple[str, ...]
    gt_sid_count: int
    decision_tokens: int
    set_nll_mean: float
    all_gt_decision_nll: float
    phase_seconds: float
    raw_gradient_norm: float
    scaled_gradient_norm: float
    rl_reference_gradient_norm: float
    lambda_effective: float
    anchor_to_rl_gradient_ratio: float
    max_replay_logp_difference: float
    optimizer_steps: int
    skip_reason: str | None


@dataclass(frozen=True)
class WindowTrainingResult:
    window_index: int
    learning_rate: float
    optimizer_updates: int
    rl_optimizer_steps: int
    anchor_optimizer_steps: int
    group_ids: tuple[str, ...]
    steps: tuple[OptimizerStepResult, ...]
    anchor: AnchorStepResult


@dataclass(frozen=True)
class AnchorGroupBackwardResult:
    set_nll: float
    all_gt_decision_nll_sum: float
    gt_sid_count: int
    decision_tokens: int
    max_replay_logp_difference: float


@dataclass(frozen=True)
class AnchorRawGradientResult:
    group_results: tuple[AnchorGroupBackwardResult, ...]
    raw_gradient_norm: float
    max_replay_logp_difference: float


# ── LR Scheduler (same as DAPO) ───────────────────────────────────────

def learning_rate_for_window(config: Mapping[str, Any], window_index: int) -> float:
    train = config["train"]
    total = int(train["total_windows"])
    warmup = int(train.get("warmup_windows", contract.WARMUP_WINDOWS))
    if not 0 <= int(window_index) < total:
        raise ValueError(f"window_index must be in [0,{total - 1}].")
    if warmup <= 0:
        raise ValueError("warmup_windows must be positive.")
    return float(train["learning_rate"]) * min(
        (int(window_index) + 1) / warmup, 1.0
    )


def _set_optimizer_lr(optimizer: torch.optim.Optimizer, value: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(value)


class WindowLRScheduler:
    """Linear warmup counted per completed logical rollout window."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: Mapping[str, Any],
        *,
        completed_windows: int = 0,
    ) -> None:
        self.optimizer = optimizer
        self.config = config
        self.completed_windows = int(completed_windows)
        total = int(config["train"]["total_windows"])
        if not 0 <= self.completed_windows <= total:
            raise ValueError("completed_windows is outside the training schedule.")
        if self.completed_windows < total:
            _set_optimizer_lr(
                optimizer,
                learning_rate_for_window(config, self.completed_windows),
            )

    def step(self) -> None:
        total = int(self.config["train"]["total_windows"])
        if self.completed_windows >= total:
            raise RuntimeError("Window scheduler is already complete.")
        self.completed_windows += 1
        if self.completed_windows < total:
            _set_optimizer_lr(
                self.optimizer,
                learning_rate_for_window(self.config, self.completed_windows),
            )

    def state_dict(self) -> dict[str, int]:
        return {"completed_windows": self.completed_windows}

    def load_state_dict(self, value: Mapping[str, Any]) -> None:
        if set(value) != {"completed_windows"}:
            raise ValueError("Window scheduler state has an invalid schema.")
        completed = value["completed_windows"]
        if isinstance(completed, bool) or not isinstance(completed, int):
            raise TypeError("completed_windows must be an integer.")
        self.completed_windows = completed
        total = int(self.config["train"]["total_windows"])
        if not 0 <= completed <= total:
            raise ValueError("Restored scheduler position is outside the schedule.")
        if completed < total:
            _set_optimizer_lr(
                self.optimizer, learning_rate_for_window(self.config, completed)
            )


# ── Partition helpers ─────────────────────────────────────────────────

def _group_key(group: SidTargetGroup) -> str:
    return f"{group.task.value}|{group.group_id}"

def partition_effective_groups(
    groups: Sequence[Any], *, window_index: int, seed: int
) -> tuple[tuple[Any, ...], ...]:
    values = tuple(groups)
    if len(values) != contract.EFFECTIVE_GROUPS_PER_WINDOW:
        raise ValueError("A complete window requires exactly 32 effective groups.")
    ids = [_group_key(item.group) for item in values]
    if len(set(ids)) != len(ids):
        raise ValueError("Effective group IDs must be unique within a window.")
    prefix = f"minibatch-order|{int(seed)}|{int(window_index)}|".encode("utf-8")
    ordered = tuple(
        sorted(
            values,
            key=lambda item: hashlib.sha256(
                prefix + _group_key(item.group).encode("utf-8")
            ).digest(),
        )
    )
    batches = tuple(
        ordered[start : start + contract.MINIBATCH_GROUPS]
        for start in range(0, len(ordered), contract.MINIBATCH_GROUPS)
    )
    if len(batches) != contract.OPTIMIZER_UPDATES_PER_WINDOW or any(
        len(batch) != contract.MINIBATCH_GROUPS for batch in batches
    ):
        raise RuntimeError("Window partition did not produce four 8-group minibatches.")
    return batches


# ── Data loading ──────────────────────────────────────────────────────

def load_groups_and_trie(
    config: Mapping[str, Any],
) -> tuple[list[SidTargetGroup], SidPrefixTrie]:
    recommendation = list(
        load_recommendation_target_groups(
            Path(config["output"]["recommendation_groups_dir"]) / "groups.jsonl"
        )
    )
    text_to_sid = list(
        load_text_to_sid_target_groups(
            Path(config["output"]["text_to_sid_groups_dir"]) / "groups.jsonl"
        )
    )
    if len(recommendation) != int(config["data"]["recommendation_groups"]):
        raise RuntimeError(
            "Recommendation group count differs from the frozen config."
        )
    if len(text_to_sid) != int(config["data"]["text_to_sid_groups"]):
        raise RuntimeError("Text-to-SID group count differs from the frozen config.")
    groups = [*recommendation, *text_to_sid]
    if len(groups) != int(config["data"]["groups"]):
        raise RuntimeError(
            f"Expected {config['data']['groups']} groups, found {len(groups)}."
        )
    trie = SidPrefixTrie.load(
        Path(config["output"]["trie_dir"]),
        expected_leaf_count=int(config["trie"]["unique_sids"]),
    )
    return groups, trie


# ── Prompt encoding ───────────────────────────────────────────────────

def _prepare_prompt(
    bundle: PolicyModel, group: SidTargetGroup, cutoff: int
) -> tuple[int, ...]:
    return tuple(
        encode_prompt(
            bundle.tokenizer,
            group.system,
            group.prompt,
            cutoff_len=int(cutoff),
        )
    )


def is_anchor_candidate(reward: RewardOutput) -> bool:
    """Whether a group should enter the GT-set anchor pool.

    Effective groups (reward has variance) and reward-flat/no-exact groups both
    enter the anchor; only reward-flat groups that already contain an exact
    rollout slot (model fully correct on this prompt) are dropped, since their
    GT-set loss is ~0 and contributes no useful gradient.
    """

    if reward.effective:
        return True
    return all(tier != "exact" for tier in reward.tiers)


# ── Effective window collection and anchor-candidate retention ────────

def collect_effective_window(
    bundle: PolicyModel,
    stream: DeterministicGroupStream,
    grammars: Mapping[SidTask, RecommendationGrammar],
    *,
    window_index: int,
    config: Mapping[str, Any],
    device: str | torch.device,
    aligned_cache: bool = True,
) -> WindowCollection:
    target = int(config["sampling"]["effective_groups_per_window"])
    raw_batch_size = int(config["rollout"]["prompt_batch_size"])
    maximum_unique = int(config["sampling"]["max_unique_prompts_per_window"])
    seen_ids: set[str] = set()
    retained: tuple[PreparedEffectiveGroup, ...] = ()
    anchor_candidates: list[PreparedFilterGroup] = []
    audits: list[RawBatchAudit] = []
    filtered_count = 0
    overflow_count = 0

    while len(retained) < target:
        remaining_unique = maximum_unique - len(seen_ids)
        if remaining_unique <= 0:
            raise RuntimeError(
                "A full dataset traversal did not produce 32 effective groups."
            )
        raw_groups = stream.take_unique(min(raw_batch_size, remaining_unique), seen_ids)
        effective_by_key: dict[str, PreparedEffectiveGroup] = {}
        anchor_by_key: dict[str, PreparedFilterGroup] = {}
        audit_parts: list[
            tuple[
                tuple[str, ...],
                tuple[str, ...],
                tuple[str, ...],
                tuple[str, ...],
                CacheRolloutStats,
            ]
        ] = []
        for task in SidTask:
            task_groups = tuple(group for group in raw_groups if group.task is task)
            if not task_groups:
                continue
            grammar = grammars[task]
            requests = tuple(
                PromptRequest(
                    group_id=_group_key(group),
                    prompt_ids=_prepare_prompt(
                        bundle, group, int(config["data"]["cutoff_len"])
                    ),
                )
                for group in task_groups
            )
            proposals, cache_stats = rollout_prompt_batch(
                bundle.model,
                requests,
                grammar,
                sampling_nonce=int(window_index),
                config=dict(config),
                device=device,
                aligned_cache=aligned_cache,
            )
            task_effective_ids: list[str] = []
            task_filtered_ids: list[str] = []
            task_anchor_ids: list[str] = []
            for group, proposal in zip(task_groups, proposals, strict=True):
                key = _group_key(group)
                canonical = canonicalize_prompt_rollout(
                    bundle.model,
                    proposal,
                    grammar,
                    sampling_nonce=int(window_index),
                    temperature=float(config["rollout"]["temperature"]),
                    max_difference=float(
                        config["rollout"]["sample_canonical_max_logp_difference"]
                    ),
                    device=device,
                )
                positives = tuple(Sid.parse(value) for value in group.positive_sids)
                reward = group_relative_rewards_and_advantages(
                    [candidate.sid for candidate in canonical.candidates], positives
                )
                if not reward.effective:
                    task_filtered_ids.append(key)
                    if is_anchor_candidate(reward):
                        task_anchor_ids.append(key)
                        anchor_by_key[key] = PreparedFilterGroup(
                            group=group,
                            prompt_ids=proposal.request.prompt_ids,
                            positives=positives,
                        )
                    continue
                task_effective_ids.append(key)
                effective_by_key[key] = PreparedEffectiveGroup(
                    group=group,
                    prompt_ids=proposal.request.prompt_ids,
                    positives=positives,
                    candidates=canonical.candidates,
                    reward=reward,
                    proposal_target_max_logp_difference=(
                        canonical.max_proposal_canonical_logp_difference
                    ),
                    corrected_candidate_count=canonical.corrected_candidate_count,
                    target_forward_calls=canonical.target_forward_calls,
                )
                task_anchor_ids.append(key)
                anchor_by_key[key] = PreparedFilterGroup(
                    group=group,
                    prompt_ids=proposal.request.prompt_ids,
                    positives=positives,
                )
            audit_parts.append(
                (
                    tuple(_group_key(group) for group in task_groups),
                    tuple(task_effective_ids),
                    tuple(task_filtered_ids),
                    tuple(task_anchor_ids),
                    cache_stats,
                )
            )

        effective = [
            effective_by_key[_group_key(group)]
            for group in raw_groups
            if _group_key(group) in effective_by_key
        ]
        anchor_candidates.extend(
            anchor_by_key[_group_key(group)]
            for group in raw_groups
            if _group_key(group) in anchor_by_key
        )

        appended = append_effective_groups(retained, effective, target=target)
        retained = appended.retained
        overflow_ids = tuple(_group_key(item.group) for item in appended.overflow)
        overflow_set = set(overflow_ids)
        filtered_count += sum(len(item[2]) for item in audit_parts)
        overflow_count += len(overflow_ids)
        for raw_ids, effective_ids, filtered_ids, anchor_ids, cache_stats in audit_parts:
            audits.append(
                RawBatchAudit(
                    raw_group_ids=raw_ids,
                    effective_group_ids=effective_ids,
                    filtered_group_ids=filtered_ids,
                    anchor_candidate_group_ids=anchor_ids,
                    overflow_group_ids=tuple(
                        key for key in effective_ids if key in overflow_set
                    ),
                    cache_stats=cache_stats,
                )
            )

    if len(retained) != target or len({_group_key(item.group) for item in retained}) != target:
        raise RuntimeError("Effective window is not exactly 32 unique groups.")

    seen_anchor: set[str] = set()
    deduped_anchor: list[PreparedFilterGroup] = []
    for item in anchor_candidates:
        gid = _group_key(item.group)
        if gid not in seen_anchor:
            seen_anchor.add(gid)
            deduped_anchor.append(item)

    return WindowCollection(
        groups=retained,
        anchor_candidates=tuple(deduped_anchor),
        source_cursor=stream.cursor,
        raw_prompt_count=len(seen_ids),
        filtered_group_count=filtered_count,
        overflow_group_count=overflow_count,
        audits=tuple(audits),
    )


# ── Optimizer / parameters ────────────────────────────────────────────

def _trainable_parameters(bundle: PolicyModel) -> list[torch.nn.Parameter]:
    values = [
        parameter for parameter in bundle.model.parameters() if parameter.requires_grad
    ]
    if not values:
        raise RuntimeError("Policy has no trainable LoRA parameters.")
    return values


def build_optimizer(
    bundle: PolicyModel, config: Mapping[str, Any]
) -> torch.optim.Optimizer:
    train = config["train"]
    kwargs: dict[str, Any] = {
        "lr": 0.0,
        "betas": (float(train["adam_beta1"]), float(train["adam_beta2"])),
        "eps": float(train["adam_epsilon"]),
        "weight_decay": float(train["weight_decay"]),
    }
    if next(bundle.model.parameters()).is_cuda:
        kwargs["fused"] = True
    return torch.optim.AdamW(_trainable_parameters(bundle), **kwargs)


def _gradient_l2_norm(parameters: Sequence[torch.nn.Parameter]) -> float:
    total: torch.Tensor | None = None
    for parameter in parameters:
        if parameter.grad is None:
            continue
        value = parameter.grad.detach().float().square().sum()
        total = value if total is None else total + value
    if total is None:
        return 0.0
    result = float(total.sqrt().item())
    if not math.isfinite(result):
        raise FloatingPointError("Gradient norm is NaN or Inf.")
    return result


def _assert_optimizer_state_step(
    optimizer: torch.optim.Optimizer, expected_step: int
) -> None:
    """Bind logged optimizer counters to AdamW's actual per-parameter step."""

    observed: set[int] = set()
    for state in optimizer.state.values():
        if "step" not in state:
            continue
        value = state["step"]
        step = int(value.item()) if isinstance(value, torch.Tensor) else int(value)
        observed.add(step)
    if int(expected_step) == 0 and not observed:
        return
    if observed != {int(expected_step)}:
        raise RuntimeError(
            "AdamW state step differs from the logged optimizer counter: "
            f"expected={expected_step}, observed={sorted(observed)}."
        )


# ── Candidate old tensor helpers (copied from DAPO) ────────────────────

def _candidate_old_tensors(
    candidates: Sequence[CanonicalCandidate],
    *,
    device: torch.device,
    completion_width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    old = torch.zeros(
        (len(candidates), int(completion_width)),
        dtype=torch.float32,
        device=device,
    )
    decisions = torch.zeros_like(old, dtype=torch.bool)
    for row, candidate in enumerate(candidates):
        length = len(candidate.old_log_probs)
        old[row, :length] = torch.tensor(
            candidate.old_log_probs, dtype=torch.float32, device=device
        )
        decisions[row, :length] = torch.tensor(
            candidate.decision_mask, dtype=torch.bool, device=device
        )
    return old, decisions


# ── RL minibatch training (Phase 1) ────────────────────────────────────

def train_minibatch(
    bundle: PolicyModel,
    optimizer: torch.optim.Optimizer,
    groups: Sequence[PreparedEffectiveGroup],
    grammars: Mapping[SidTask, RecommendationGrammar],
    *,
    config: Mapping[str, Any],
    device: str | torch.device,
    enforce_replay_gate: bool,
) -> OptimizerStepResult:
    values = tuple(groups)
    if len(values) != contract.MINIBATCH_GROUPS:
        raise ValueError("An optimizer minibatch requires exactly eight groups.")
    torch_device = torch.device(device)
    total_decisions = sum(
        sum(candidate.decision_mask)
        for group in values
        for candidate in group.candidates
    )
    if total_decisions <= 0:
        raise RuntimeError("Optimizer minibatch contains no decision token.")
    parameters = _trainable_parameters(bundle)
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    maximum_delta = 0.0
    ratios: list[float] = []
    clipped_tokens = 0
    below_tokens = 0
    above_tokens = 0
    replay_limit = float(config["rollout"]["canonical_replay_max_logp_difference"])
    for prepared in values:
        grammar = grammars[prepared.group.task]
        scoring_width = grammar_completion_width(grammar)
        layer_advantages = prepared.reward.layer_advantages.to(torch_device)
        for start in (0, contract.LOSS_CHUNK_SIZE):
            chunk = prepared.candidates[start : start + contract.LOSS_CHUNK_SIZE]
            scores = score_completions_dense_fixed(
                bundle.model,
                prepared.prompt_ids,
                [candidate.token_ids for candidate in chunk],
                grammar,
                device=torch_device,
                temperature=float(config["rollout"]["temperature"]),
                batch_rows=contract.LOSS_CHUNK_SIZE,
                completion_width=scoring_width,
            )
            old, expected_mask = _candidate_old_tensors(
                chunk, device=torch_device, completion_width=scoring_width
            )
            if not torch.equal(scores.decision_mask, expected_mask):
                raise RuntimeError("Canonical old/new decision masks differ.")
            delta = (scores.log_probs - old).abs()[scores.decision_mask]
            chunk_delta = float(delta.max().detach().item()) if delta.numel() else 0.0
            maximum_delta = max(maximum_delta, chunk_delta)
            if enforce_replay_gate and chunk_delta > replay_limit:
                raise RuntimeError(
                    "Canonical replay probability gate failed before update: "
                    f"max_abs_delta={chunk_delta:.9g}, limit={replay_limit:.9g}."
                )
            output = clipped_group_relative_token_sum(
                scores.log_probs,
                old,
                layer_advantages[start : start + contract.LOSS_CHUNK_SIZE],
                scores.decision_mask,
                clip_low=float(config["loss"]["clip_ratio_low"]),
                clip_high=float(config["loss"]["clip_ratio_high"]),
            )
            (output.loss_sum / total_decisions).backward()
            loss_sum += float(output.loss_sum.detach().item())
            ratios.extend(float(value) for value in output.ratios.cpu().tolist())
            clipped_tokens += int(output.clip_active.sum().item())
            below_tokens += int(output.below_low.sum().item())
            above_tokens += int(output.above_high.sum().item())
            del scores, old, expected_mask, output

    gradient_norm = _gradient_l2_norm(parameters)
    torch.nn.utils.clip_grad_norm_(
        parameters, float(config["train"]["max_grad_norm"])
    )
    optimizer.step()
    if not ratios or not all(math.isfinite(value) and value > 0.0 for value in ratios):
        raise FloatingPointError(
            "Optimizer minibatch ratios are not positive finite values."
        )
    return OptimizerStepResult(
        loss=loss_sum / total_decisions,
        learning_rate=float(optimizer.param_groups[0]["lr"]),
        decision_tokens=total_decisions,
        gradient_norm=gradient_norm,
        max_replay_logp_difference=maximum_delta,
        ratio_min=min(ratios),
        ratio_max=max(ratios),
        ratio_mean=sum(ratios) / len(ratios),
        clipped_tokens=clipped_tokens,
        below_low_tokens=below_tokens,
        above_high_tokens=above_tokens,
        group_ids=tuple(_group_key(group.group) for group in values),
    )


# ── Anchor phase (Phase 2) ────────────────────────────────────────────

def _encode_anchor_completions(
    group: PreparedFilterGroup, grammar: RecommendationGrammar
) -> tuple[tuple[int, ...], ...]:
    if not group.positives:
        raise RuntimeError("An anchor candidate has no GT SID.")
    completions: list[tuple[int, ...]] = []
    for sid in group.positives:
        try:
            completion = tuple(grammar.encode_sid(sid))
        except Exception as exc:
            raise RuntimeError(
                f"Anchor GT SID is absent from the frozen trie: {sid.render()}"
            ) from exc
        completions.append(completion)
    if len(set(completions)) != len(completions):
        raise RuntimeError("Distinct GT SIDs encoded to duplicate completion paths.")
    return tuple(completions)


@dataclass(frozen=True)
class AnchorCompletionRow:
    group_index: int
    gt_index: int
    task: SidTask
    group_id: str
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]


@dataclass(frozen=True)
class AnchorReplayBatch:
    task: SidTask
    rows: tuple[AnchorCompletionRow, ...]
    first_logps: torch.Tensor
    first_decision_mask: torch.Tensor
    first_decision_counts: torch.Tensor


def build_anchor_raw_gradient(
    bundle: PolicyModel,
    optimizer: torch.optim.Optimizer,
    anchor_candidates: Sequence[PreparedFilterGroup],
    grammars: Mapping[SidTask, RecommendationGrammar],
    *,
    config: Mapping[str, Any],
    device: str | torch.device,
) -> AnchorRawGradientResult:
    """Average every eligible group into one raw, unweighted anchor gradient."""

    candidates = tuple(anchor_candidates)
    if not candidates:
        raise ValueError("Raw anchor gradient requires at least one candidate.")
    ids = tuple(_group_key(item.group) for item in candidates)
    if len(set(ids)) != len(ids):
        raise RuntimeError("Anchor candidates must be unique within a window.")
    microbatch = int(config["anchor"]["gt_completion_microbatch"])
    if microbatch != contract.ANCHOR_GT_COMPLETION_MICROBATCH:
        raise ValueError("V2.2 fixes the GT completion microbatch at eight rows.")
    torch_device = torch.device(device)
    rows: list[AnchorCompletionRow] = []
    completion_counts: list[int] = []
    for group_index, candidate in enumerate(candidates):
        grammar = grammars[candidate.group.task]
        completions = _encode_anchor_completions(candidate, grammar)
        completion_counts.append(len(completions))
        rows.extend(
            AnchorCompletionRow(
                group_index=group_index,
                gt_index=gt_index,
                task=candidate.group.task,
                group_id=_group_key(candidate.group),
                prompt_ids=candidate.prompt_ids,
                completion_ids=completion,
            )
            for gt_index, completion in enumerate(completions)
        )
    sequence_by_group = [torch.empty(count, dtype=torch.float32) for count in completion_counts]
    decision_by_group = [torch.empty(count, dtype=torch.long) for count in completion_counts]
    replay_batches: list[AnchorReplayBatch] = []
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for task in SidTask:
            ordered = tuple(
                sorted(
                    (row for row in rows if row.task is task),
                    key=lambda row: (
                        len(row.prompt_ids),
                        row.group_id,
                        row.gt_index,
                    ),
                )
            )
            grammar = grammars[task]
            width = grammar_completion_width(grammar)
            for start in range(0, len(ordered), microbatch):
                batch_rows = ordered[start : start + microbatch]
                scores = score_prompt_completion_pairs_dense(
                    bundle.model,
                    [row.prompt_ids for row in batch_rows],
                    [row.completion_ids for row in batch_rows],
                    grammar,
                    device=torch_device,
                    temperature=float(config["anchor"]["teacher_forcing_temperature"]),
                    completion_width=width,
                    max_batch_rows=microbatch,
                )
                sequences, counts = gt_sequence_logps(
                    scores.log_probs, scores.decision_mask
                )
                for row_index, row in enumerate(batch_rows):
                    sequence_by_group[row.group_index][row.gt_index] = sequences[
                        row_index
                    ].detach().cpu()
                    decision_by_group[row.group_index][row.gt_index] = counts[
                        row_index
                    ].detach().cpu()
                replay_batches.append(
                    AnchorReplayBatch(
                        task=task,
                        rows=tuple(batch_rows),
                        first_logps=scores.log_probs.detach().cpu(),
                        first_decision_mask=scores.decision_mask.detach().cpu(),
                        first_decision_counts=counts.detach().cpu(),
                    )
                )
                del scores, sequences, counts

    results: list[AnchorGroupBackwardResult] = []
    weights_by_group: list[torch.Tensor] = []
    for sequences, counts in zip(sequence_by_group, decision_by_group, strict=True):
        results.append(
            AnchorGroupBackwardResult(
                set_nll=float(gt_set_anchor_loss(sequences).item()),
                all_gt_decision_nll_sum=float((-sequences.sum()).item()),
                gt_sid_count=sequences.numel(),
                decision_tokens=int(counts.sum().item()),
                max_replay_logp_difference=0.0,
            )
        )
        weights_by_group.append(sequences.softmax(dim=0))

    maximum_delta = 0.0
    limit = float(config["anchor"]["replay_max_logp_difference"])
    for replay in replay_batches:
        grammar = grammars[replay.task]
        width = grammar_completion_width(grammar)
        scores = score_prompt_completion_pairs_dense(
            bundle.model,
            [row.prompt_ids for row in replay.rows],
            [row.completion_ids for row in replay.rows],
            grammar,
            device=torch_device,
            temperature=float(config["anchor"]["teacher_forcing_temperature"]),
            completion_width=width,
            max_batch_rows=microbatch,
        )
        sequences, counts = gt_sequence_logps(scores.log_probs, scores.decision_mask)
        current_mask = scores.decision_mask.detach().cpu()
        if not torch.equal(current_mask, replay.first_decision_mask):
            raise RuntimeError("Anchor no-grad/grad decision masks differ.")
        if not torch.equal(counts.detach().cpu(), replay.first_decision_counts):
            raise RuntimeError("Anchor no-grad/grad decision counts differ.")
        delta = (scores.log_probs.detach().cpu() - replay.first_logps).abs()[
            replay.first_decision_mask
        ]
        if delta.numel():
            maximum_delta = max(maximum_delta, float(delta.max().item()))
        if maximum_delta > limit:
            raise RuntimeError(
                "Anchor no-grad/grad probability gate failed: "
                f"max_abs_delta={maximum_delta:.9g}, limit={limit:.9g}."
            )
        row_weights = torch.stack(
            [
                weights_by_group[row.group_index][row.gt_index]
                for row in replay.rows
            ]
        ).to(torch_device)
        surrogate = -(
            row_weights * sequences
        ).sum() / float(len(candidates))
        if surrogate.requires_grad:
            surrogate.backward()
        del scores, sequences, counts, surrogate

    results = [
        AnchorGroupBackwardResult(
            set_nll=result.set_nll,
            all_gt_decision_nll_sum=result.all_gt_decision_nll_sum,
            gt_sid_count=result.gt_sid_count,
            decision_tokens=result.decision_tokens,
            max_replay_logp_difference=maximum_delta,
        )
        for result in results
    ]
    return AnchorRawGradientResult(
        group_results=tuple(results),
        raw_gradient_norm=_gradient_l2_norm(_trainable_parameters(bundle)),
        max_replay_logp_difference=maximum_delta,
    )


def _scale_gradients(parameters: Sequence[torch.nn.Parameter], scale: float) -> None:
    if not math.isfinite(float(scale)) or float(scale) < 0.0:
        raise ValueError("Gradient scale must be finite and non-negative.")
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.mul_(float(scale))


def _empty_anchor_result(
    *,
    candidate_count: int,
    reason: str,
    rl_reference_gradient_norm: float = 0.0,
) -> AnchorStepResult:
    return AnchorStepResult(
        candidate_count=int(candidate_count),
        groups_used=0,
        group_ids=(),
        gt_sid_count=0,
        decision_tokens=0,
        set_nll_mean=0.0,
        all_gt_decision_nll=0.0,
        phase_seconds=0.0,
        raw_gradient_norm=0.0,
        scaled_gradient_norm=0.0,
        rl_reference_gradient_norm=float(rl_reference_gradient_norm),
        lambda_effective=0.0,
        anchor_to_rl_gradient_ratio=0.0,
        max_replay_logp_difference=0.0,
        optimizer_steps=0,
        skip_reason=reason,
    )


def run_anchor_phase(
    bundle: PolicyModel,
    optimizer: torch.optim.Optimizer,
    anchor_candidates: Sequence[PreparedFilterGroup],
    grammars: Mapping[SidTask, RecommendationGrammar],
    *,
    rl_reference_gradient_norm: float,
    config: Mapping[str, Any],
    device: str | torch.device,
) -> AnchorStepResult:
    """Use all candidates, then take zero or one bounded anchor optimizer step."""

    candidates = tuple(anchor_candidates)
    if not candidates:
        return _empty_anchor_result(
            candidate_count=0,
            reason="no_anchor_candidates",
        )
    started = time.perf_counter()
    raw = build_anchor_raw_gradient(
        bundle,
        optimizer,
        candidates,
        grammars,
        config=config,
        device=device,
    )
    parameters = _trainable_parameters(bundle)
    budget: AnchorLambdaOutput = anchor_effective_lambda(
        rl_reference_grad_norm=float(rl_reference_gradient_norm),
        anchor_raw_grad_norm=float(raw.raw_gradient_norm),
        target_gradient_ratio=float(config["anchor"]["target_gradient_ratio"]),
        epsilon=float(config["anchor"]["gradient_epsilon"]),
    )
    _scale_gradients(parameters, budget.lambda_effective)
    scaled_norm = _gradient_l2_norm(parameters)
    ratio = (
        scaled_norm / float(rl_reference_gradient_norm)
        if float(rl_reference_gradient_norm) > 0.0
        else 0.0
    )
    target = float(config["anchor"]["target_gradient_ratio"])
    limit = target * float(rl_reference_gradient_norm)
    tolerance = max(1.0e-12, abs(limit) * 1.0e-6)
    if scaled_norm > limit + tolerance:
        raise RuntimeError(
            "Scaled anchor gradient exceeds the pre-clip RL budget: "
            f"scaled={scaled_norm:.9g}, limit={limit:.9g}."
        )

    optimizer_steps = 0
    skip_reason: str | None = None
    if budget.update_allowed:
        torch.nn.utils.clip_grad_norm_(
            parameters, float(config["train"]["max_grad_norm"])
        )
        optimizer.step()
        optimizer_steps = 1
    else:
        if float(rl_reference_gradient_norm) <= 0.0:
            skip_reason = "zero_rl_reference_gradient"
        else:
            skip_reason = "zero_anchor_raw_gradient"
        optimizer.zero_grad(set_to_none=True)

    results = raw.group_results
    total_decisions = sum(result.decision_tokens for result in results)
    all_gt_nll_sum = sum(result.all_gt_decision_nll_sum for result in results)
    return AnchorStepResult(
        candidate_count=len(candidates),
        groups_used=len(candidates),
        group_ids=tuple(_group_key(item.group) for item in candidates),
        gt_sid_count=sum(result.gt_sid_count for result in results),
        decision_tokens=total_decisions,
        set_nll_mean=sum(result.set_nll for result in results) / len(results),
        all_gt_decision_nll=(
            all_gt_nll_sum / total_decisions if total_decisions else 0.0
        ),
        phase_seconds=time.perf_counter() - started,
        raw_gradient_norm=raw.raw_gradient_norm,
        scaled_gradient_norm=scaled_norm,
        rl_reference_gradient_norm=float(rl_reference_gradient_norm),
        lambda_effective=budget.lambda_effective,
        anchor_to_rl_gradient_ratio=ratio,
        max_replay_logp_difference=raw.max_replay_logp_difference,
        optimizer_steps=optimizer_steps,
        skip_reason=skip_reason,
    )


# ── Complete window training ──────────────────────────────────────────

def train_complete_window(
    bundle: PolicyModel,
    optimizer: torch.optim.Optimizer,
    scheduler: WindowLRScheduler,
    groups: Sequence[PreparedEffectiveGroup],
    anchor_candidates: Sequence[PreparedFilterGroup],
    grammars: Mapping[SidTask, RecommendationGrammar],
    *,
    window_index: int,
    config: Mapping[str, Any],
    device: str | torch.device,
) -> WindowTrainingResult:
    if scheduler.completed_windows != int(window_index):
        raise RuntimeError("Scheduler and requested window index differ.")
    expected_lr = learning_rate_for_window(config, window_index)
    if any(
        not math.isclose(float(group["lr"]), expected_lr, rel_tol=0.0, abs_tol=1e-15)
        for group in optimizer.param_groups
    ):
        raise RuntimeError("Optimizer LR differs from the window schedule.")

    # Phase 1: RL (4 optimizer steps)
    minibatches = partition_effective_groups(
        groups,
        window_index=int(window_index),
        seed=int(config["train"]["seed"]),
    )
    steps: list[OptimizerStepResult] = []
    for minibatch_index, minibatch in enumerate(minibatches):
        step = train_minibatch(
            bundle,
            optimizer,
            minibatch,
            grammars,
            config=config,
            device=device,
            enforce_replay_gate=minibatch_index == 0,
        )
        steps.append(step)
        print(
            json.dumps(
                {
                    "event": "optimizer_step_complete",
                    "window_index": int(window_index),
                    "minibatch_index": minibatch_index,
                    "loss": step.loss,
                    "learning_rate": step.learning_rate,
                    "ratio_min": step.ratio_min,
                    "ratio_max": step.ratio_max,
                    "clipped_tokens": step.clipped_tokens,
                },
                separators=(",", ":"),
                allow_nan=False,
            ),
            flush=True,
        )
    rl_reference_gradient_norm = float(median(step.gradient_norm for step in steps))

    # Phase 2: from W0, at most one independent anchor update at this window's LR.
    if anchor_candidates:
        anchor_result = run_anchor_phase(
            bundle,
            optimizer,
            anchor_candidates,
            grammars,
            rl_reference_gradient_norm=rl_reference_gradient_norm,
            config=config,
            device=device,
        )
        print(
            json.dumps(
                {
                    "event": "anchor_phase_complete",
                    "window_index": int(window_index),
                    "anchor_candidate_groups": anchor_result.candidate_count,
                    "anchor_groups_used": anchor_result.groups_used,
                    "anchor_set_nll_mean": anchor_result.set_nll_mean,
                    "anchor_raw_grad_norm": anchor_result.raw_gradient_norm,
                    "lambda_effective": anchor_result.lambda_effective,
                    "anchor_optimizer_steps": anchor_result.optimizer_steps,
                },
                separators=(",", ":"),
                allow_nan=False,
            ),
            flush=True,
        )
    else:
        anchor_result = _empty_anchor_result(
            candidate_count=0,
            reason="no_anchor_candidates",
            rl_reference_gradient_norm=rl_reference_gradient_norm,
        )
    scheduler.step()

    ids = tuple(group_id for step in steps for group_id in step.group_ids)
    if len(ids) != contract.EFFECTIVE_GROUPS_PER_WINDOW or len(set(ids)) != len(ids):
        raise RuntimeError("K=1 invariant failed: a group was lost or reused.")

    return WindowTrainingResult(
        window_index=int(window_index),
        learning_rate=expected_lr,
        optimizer_updates=len(steps) + anchor_result.optimizer_steps,
        rl_optimizer_steps=len(steps),
        anchor_optimizer_steps=anchor_result.optimizer_steps,
        group_ids=ids,
        steps=tuple(steps),
        anchor=anchor_result,
    )


# ── Determinism utilities ─────────────────────────────────────────────

def set_global_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def configure_deterministic_runtime(seed: int) -> None:
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    set_global_seed(seed)


# ── JSON helpers ──────────────────────────────────────────────────────

def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _append_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                json.dumps(
                    dict(row),
                    ensure_ascii=False,
                    separators=(",", ":"),
                    allow_nan=False,
                )
            )
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _truncate_jsonl(path: Path, rows: int) -> None:
    if rows < 0:
        raise ValueError("JSONL row count must be non-negative.")
    if not path.exists():
        if rows:
            raise RuntimeError(
                f"Recovery expects {rows} rows but {path} is missing."
            )
        return
    raw_lines = path.read_bytes().splitlines(keepends=True)
    if len(raw_lines) < rows:
        raise RuntimeError(
            f"Recovery expects {rows} rows but {path} contains {len(raw_lines)}."
        )
    for line_number, raw in enumerate(raw_lines[:rows], start=1):
        try:
            json.loads(raw)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Corrupt JSONL row {line_number}: {path}"
            ) from exc
    if len(raw_lines) != rows or (
        path.stat().st_size and not raw_lines[-1].endswith(b"\n")
    ):
        temporary = path.with_name(f".{path.name}.truncate")
        temporary.write_bytes(b"".join(raw_lines[:rows]))
        os.replace(temporary, path)


# ── Logging ───────────────────────────────────────────────────────────

def _group_log_rows(
    collection: WindowCollection,
    *,
    window_index: int,
    anchor_result: AnchorStepResult,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    for prepared in collection.groups:
        rows.append(
            {
                "schema_version": 2,
                "window_index": int(window_index),
                "effective_epoch": int(window_index)
                // contract.WINDOWS_PER_EFFECTIVE_EPOCH,
                "task": prepared.group.task.value,
                "group_id": prepared.group.group_id,
                "group_key": _group_key(prepared.group),
                "source_lines": list(prepared.group.source_lines),
                "positive_sids": list(prepared.group.positive_sids),
                "candidate_indices": [
                    candidate.candidate_index for candidate in prepared.candidates
                ],
                "candidate_sids": [
                    candidate.sid.render() for candidate in prepared.candidates
                ],
                "candidate_token_ids": [
                    list(candidate.token_ids) for candidate in prepared.candidates
                ],
                "rewards": prepared.reward.rewards.tolist(),
                "reward_tiers": list(prepared.reward.tiers),
                "advantages": prepared.reward.advantages.tolist(),
                "effective": prepared.reward.effective,
                "proposal_target_max_logp_difference": (
                    prepared.proposal_target_max_logp_difference
                ),
                "corrected_candidate_count": prepared.corrected_candidate_count,
                "target_forward_calls": prepared.target_forward_calls,
                "anchor_candidate_groups": anchor_result.candidate_count,
                "anchor_groups_used": anchor_result.groups_used,
                "gt_injection_count": 0,
                "k": contract.NUM_ITERATIONS,
            }
        )
    return rows


def _window_log_row(
    collection: WindowCollection,
    training: WindowTrainingResult,
    *,
    collection_seconds: float,
    training_seconds: float,
    peak_reserved_gib: float,
    collection_peak_reserved_gib: float,
    training_peak_reserved_gib: float,
    rl_optimizer_update_step: int,
    anchor_optimizer_update_step: int,
    optimizer_update_step: int,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "window_index": training.window_index,
        "effective_epoch": (
            training.window_index // contract.WINDOWS_PER_EFFECTIVE_EPOCH
        ),
        "learning_rate": training.learning_rate,
        "effective_groups": len(collection.groups),
        "filter_groups": collection.filtered_group_count,
        "rollouts": len(collection.groups) * contract.GROUP_SIZE,
        "raw_prompt_count": collection.raw_prompt_count,
        "filtered_group_count": collection.filtered_group_count,
        "overflow_group_count": collection.overflow_group_count,
        "anchor_candidate_groups": training.anchor.candidate_count,
        "anchor_groups_used": training.anchor.groups_used,
        "anchor_group_ids": list(training.anchor.group_ids),
        "anchor_gt_sid_count": training.anchor.gt_sid_count,
        "anchor_decision_token_count": training.anchor.decision_tokens,
        "anchor_set_nll_mean": training.anchor.set_nll_mean,
        "anchor_all_gt_decision_nll": training.anchor.all_gt_decision_nll,
        "anchor_phase_seconds": training.anchor.phase_seconds,
        "anchor_raw_grad_norm": training.anchor.raw_gradient_norm,
        "anchor_post_scale_grad_norm": training.anchor.scaled_gradient_norm,
        "rl_reference_grad_norm": training.anchor.rl_reference_gradient_norm,
        "lambda_effective": training.anchor.lambda_effective,
        "anchor_to_rl_grad_ratio": training.anchor.anchor_to_rl_gradient_ratio,
        "anchor_max_replay_logp_difference": (
            training.anchor.max_replay_logp_difference
        ),
        "anchor_skip_reason": training.anchor.skip_reason,
        "group_ids": list(training.group_ids),
        "rl_optimizer_steps": training.rl_optimizer_steps,
        "anchor_optimizer_steps": training.anchor_optimizer_steps,
        "optimizer_updates": training.optimizer_updates,
        "rl_optimizer_update_step": int(rl_optimizer_update_step),
        "anchor_optimizer_update_step": int(anchor_optimizer_update_step),
        "optimizer_update_step": int(optimizer_update_step),
        "steps": [asdict(step) for step in training.steps],
        "source_cursor": asdict(collection.source_cursor),
        "sampling_batches": [
            {
                **asdict(audit),
                "cache_stats": asdict(audit.cache_stats),
            }
            for audit in collection.audits
        ],
        "collection_seconds": float(collection_seconds),
        "training_seconds": float(training_seconds),
        "window_seconds": float(collection_seconds + training_seconds),
        "peak_reserved_gib": float(peak_reserved_gib),
        "collection_peak_reserved_gib": float(collection_peak_reserved_gib),
        "training_peak_reserved_gib": float(training_peak_reserved_gib),
        "gt_injection_count": 0,
        "k": contract.NUM_ITERATIONS,
    }


# ── Run directory management ──────────────────────────────────────────

def _write_run_contract(
    output_dir: Path,
    config: Mapping[str, Any],
    signature: Mapping[str, Any],
) -> None:
    signature_path = output_dir / "runtime_signature.json"
    config_path = output_dir / "resolved_config.json"
    if signature_path.exists() or config_path.exists():
        raise RuntimeError("A new run directory already contains contract files.")
    _atomic_json(signature_path, signature)
    _atomic_json(config_path, config)


def _read_run_contract(path: Path, *, label: str) -> Any:
    if not path.is_file():
        raise RuntimeError(f"Existing run directory is missing {label}.")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Existing run directory contains unreadable {label}."
        ) from exc


def _prepare_run_directory(
    output_dir: Path,
    config: Mapping[str, Any],
    signature: Mapping[str, Any],
    *,
    resume: bool,
) -> None:
    if output_dir.exists():
        if not output_dir.is_dir():
            raise FileExistsError(f"Output path is not a directory: {output_dir}")
        is_non_empty = next(output_dir.iterdir(), None) is not None
        if is_non_empty:
            if not resume:
                raise FileExistsError(f"Non-empty output directory: {output_dir}")
            existing_signature = _read_run_contract(
                output_dir / "runtime_signature.json",
                label="runtime_signature.json",
            )
            existing_config = _read_run_contract(
                output_dir / "resolved_config.json",
                label="resolved_config.json",
            )
            if existing_signature != signature or existing_config != config:
                raise RuntimeError(
                    "Existing run contract differs from this invocation."
                )
            return
    else:
        output_dir.mkdir(parents=True)

    _write_run_contract(output_dir, config, signature)


def _ensure_epoch_adapters(
    bundle: PolicyModel,
    output: Path,
    *,
    completed_windows: int,
) -> None:
    completed_epochs = completed_windows // contract.WINDOWS_PER_EFFECTIVE_EPOCH
    for epoch_number in range(1, completed_epochs + 1):
        epoch_dir = output / f"epoch-{epoch_number:02d}-adapter"
        if epoch_dir.exists():
            continue
        boundary = epoch_number * contract.WINDOWS_PER_EFFECTIVE_EPOCH
        if completed_windows != boundary:
            raise RuntimeError(
                f"Missing epoch-{epoch_number:02d} adapter cannot be "
                f"reconstructed from window {completed_windows}."
            )
        save_policy_atomic(bundle, epoch_dir)


# ── Training impl ─────────────────────────────────────────────────────

def _run_training_impl(
    config: dict[str, Any],
    runtime_signature: dict[str, Any],
    *,
    output_dir: Path,
    device: str | torch.device = "cuda:0",
    resume: bool = True,
    stop_after_windows: int | None = None,
    aligned_cache: bool = True,
) -> dict[str, Any]:
    validate_runtime_signature(runtime_signature)
    output = Path(output_dir).expanduser().resolve()
    _prepare_run_directory(output, config, runtime_signature, resume=resume)
    configure_deterministic_runtime(int(config["train"]["seed"]))
    recovery_root = output / "recovery"
    recovery = (
        load_latest_recovery(recovery_root, runtime_signature)
        if resume
        else None
    )
    if recovery is None:
        cursor = RecoveryCursor(
            next_window_index=0,
            rl_optimizer_update_step=0,
            anchor_optimizer_update_step=0,
            optimizer_update_step=0,
            source_cursor=SourceCursor(cycle_index=0, offset=0),
            window_log_rows=0,
            group_log_rows=0,
        )
        policy_path = None
        training_state = None
    else:
        checkpoint_path, cursor, training_state = recovery
        policy_path = checkpoint_path

    window_log = output / "windows.jsonl"
    group_log = output / "groups.jsonl"
    _truncate_jsonl(window_log, cursor.window_log_rows)
    _truncate_jsonl(group_log, cursor.group_log_rows)

    groups, trie = load_groups_and_trie(config)
    bundle = load_policy_model(
        config, device=device, policy_adapter_path=policy_path
    )
    _ensure_epoch_adapters(
        bundle,
        output,
        completed_windows=cursor.next_window_index,
    )
    grammars = {
        SidTask.RECOMMENDATION: RecommendationGrammar(
            bundle.tokenizer, trie, mode="train_recommendation"
        ),
        SidTask.ITEM_TEXT_TO_SID: RecommendationGrammar(
            bundle.tokenizer, trie, mode="train_text_to_sid"
        ),
    }
    optimizer = build_optimizer(bundle, config)
    scheduler = WindowLRScheduler(optimizer, config)
    if training_state is not None:
        restore_training_state(optimizer, scheduler, training_state)
    if scheduler.completed_windows != cursor.next_window_index:
        raise RuntimeError("Recovery scheduler and cursor window differ.")
    _assert_optimizer_state_step(optimizer, cursor.optimizer_update_step)
    stream = DeterministicGroupStream(
        groups,
        seed=int(config["train"]["seed"]),
        cursor=cursor.source_cursor,
    )

    total_windows = int(config["train"]["total_windows"])
    requested_stop = (
        total_windows
        if stop_after_windows is None
        else min(
            total_windows,
            cursor.next_window_index + int(stop_after_windows),
        )
    )
    if requested_stop < cursor.next_window_index:
        raise ValueError("stop_after_windows must be non-negative.")
    last_checkpoint: Path | None = (
        recovery[0] if recovery is not None else None
    )
    windows_run = 0

    for window_index in range(cursor.next_window_index, requested_stop):
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch.device(device))
        start = time.perf_counter()

        # Collect 32 effective RL groups and every reward-flat/no-exact anchor candidate.
        collection = collect_effective_window(
            bundle,
            stream,
            grammars,
            window_index=window_index,
            config=config,
            device=device,
            aligned_cache=aligned_cache,
        )
        collection_seconds = time.perf_counter() - start
        print(
            json.dumps(
                {
                    "event": "collection_complete",
                    "window_index": window_index,
                    "raw_prompt_count": collection.raw_prompt_count,
                    "effective_groups": len(collection.groups),
                    "filter_groups": collection.filtered_group_count,
                    "anchor_candidate_groups": len(collection.anchor_candidates),
                    "filtered_groups": collection.filtered_group_count,
                    "seconds": collection_seconds,
                },
                separators=(",", ":"),
                allow_nan=False,
            ),
            flush=True,
        )
        collection_peak_reserved = (
            torch.cuda.max_memory_reserved(torch.device(device)) / 1024**3
            if torch.cuda.is_available() and torch.device(device).type == "cuda"
            else 0.0
        )
        if torch.cuda.is_available() and torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(torch.device(device))

        start = time.perf_counter()

        # 训练: Phase 1 (RL) + Phase 2 (Anchor)
        training = train_complete_window(
            bundle,
            optimizer,
            scheduler,
            collection.groups,
            collection.anchor_candidates,
            grammars,
            window_index=window_index,
            config=config,
            device=device,
        )
        training_seconds = time.perf_counter() - start
        training_peak_reserved = (
            torch.cuda.max_memory_reserved(torch.device(device)) / 1024**3
            if torch.cuda.is_available() and torch.device(device).type == "cuda"
            else 0.0
        )
        peak_reserved = max(collection_peak_reserved, training_peak_reserved)
        if peak_reserved > float(config["memory"]["max_reserved_gib"]):
            raise RuntimeError(
                f"Window reserved memory {peak_reserved:.3f} GiB exceeds limit."
            )

        next_rl_update = (
            cursor.rl_optimizer_update_step + training.rl_optimizer_steps
        )
        next_anchor_update = (
            cursor.anchor_optimizer_update_step + training.anchor_optimizer_steps
        )
        next_optimizer_update = next_rl_update + next_anchor_update
        _assert_optimizer_state_step(optimizer, next_optimizer_update)

        # Logs are appended only after the whole window, including anchor, succeeds.
        _append_jsonl(
            group_log,
            _group_log_rows(
                collection,
                window_index=window_index,
                anchor_result=training.anchor,
            ),
        )
        _append_jsonl(
            window_log,
            [
                _window_log_row(
                    collection,
                    training,
                    collection_seconds=collection_seconds,
                    training_seconds=training_seconds,
                    peak_reserved_gib=peak_reserved,
                    collection_peak_reserved_gib=collection_peak_reserved,
                    training_peak_reserved_gib=training_peak_reserved,
                    rl_optimizer_update_step=next_rl_update,
                    anchor_optimizer_update_step=next_anchor_update,
                    optimizer_update_step=next_optimizer_update,
                )
            ],
        )

        # Checkpoint
        next_window = window_index + 1
        cursor = RecoveryCursor(
            next_window_index=next_window,
            rl_optimizer_update_step=next_rl_update,
            anchor_optimizer_update_step=next_anchor_update,
            optimizer_update_step=next_optimizer_update,
            source_cursor=collection.source_cursor,
            window_log_rows=next_window,
            group_log_rows=next_window * contract.EFFECTIVE_GROUPS_PER_WINDOW,
        )
        epoch_finished = next_window % contract.WINDOWS_PER_EFFECTIVE_EPOCH == 0
        should_checkpoint = (
            recovery_checkpoint_due(
                cursor,
                epoch_finished=epoch_finished,
                interval_windows=int(
                    config["train"]["resume_save_windows"]
                ),
            )
            or next_window == requested_stop
        )
        if should_checkpoint:
            checkpoint_path = recovery_root / (
                f"checkpoint-window-{cursor.next_window_index:06d}"
                f"-update-{cursor.optimizer_update_step:06d}"
            )
            if checkpoint_path.exists():
                raise FileExistsError(checkpoint_path)
            last_checkpoint = save_recovery_checkpoint(
                bundle,
                optimizer,
                scheduler,
                cursor,
                recovery_root,
                runtime_signature,
            )
        if epoch_finished:
            epoch_number = next_window // contract.WINDOWS_PER_EFFECTIVE_EPOCH
            epoch_dir = output / f"epoch-{epoch_number:02d}-adapter"
            if not epoch_dir.exists():
                save_policy_atomic(bundle, epoch_dir)
        print(
            json.dumps(
                {
                    "event": "window_complete",
                    "completed_windows": next_window,
                    "rl_optimizer_update_step": cursor.rl_optimizer_update_step,
                    "anchor_optimizer_update_step": (
                        cursor.anchor_optimizer_update_step
                    ),
                    "optimizer_update_step": cursor.optimizer_update_step,
                    "window_seconds": collection_seconds + training_seconds,
                    "peak_reserved_gib": peak_reserved,
                    "checkpoint_saved": bool(should_checkpoint),
                    "anchor_candidate_groups": training.anchor.candidate_count,
                    "anchor_groups_used": training.anchor.groups_used,
                    "anchor_set_nll_mean": training.anchor.set_nll_mean,
                    "lambda_effective": training.anchor.lambda_effective,
                },
                separators=(",", ":"),
                allow_nan=False,
            ),
            flush=True,
        )
        windows_run += 1

    if cursor.next_window_index == total_windows:
        final_dir = output / "final-adapter"
        if not final_dir.exists():
            save_policy_atomic(bundle, final_dir)

    summary = {
        "schema_version": 2,
        "completed_windows": cursor.next_window_index,
        "rl_optimizer_update_step": cursor.rl_optimizer_update_step,
        "anchor_optimizer_update_step": cursor.anchor_optimizer_update_step,
        "optimizer_update_step": cursor.optimizer_update_step,
        "windows_run_this_invocation": windows_run,
        "source_cursor": asdict(cursor.source_cursor),
        "last_checkpoint": (
            last_checkpoint.relative_to(output).as_posix()
            if last_checkpoint
            else None
        ),
        "complete": cursor.next_window_index == total_windows,
        "gt_injection_count": 0,
        "k": contract.NUM_ITERATIONS,
    }
    _atomic_json(output / "run_summary.json", summary)
    return summary


def run_training(
    config: dict[str, Any],
    runtime_signature: dict[str, Any],
    *,
    output_dir: Path,
    device: str | torch.device = "cuda:0",
    resume: bool = True,
    stop_after_windows: int | None = None,
    aligned_cache: bool = True,
    dense_scoring: bool = False,
    formal: bool = False,
) -> dict[str, Any]:
    if formal:
        expected_output = (
            Path(config["output"]["run_dir"]).expanduser().resolve()
        )
        actual_output = Path(output_dir).expanduser().resolve()
        if actual_output != expected_output:
            raise ValueError(
                "Formal training output_dir must equal config.output.run_dir."
            )
        if stop_after_windows is not None:
            raise ValueError(
                "Formal training cannot set stop_after_windows."
            )
        if dense_scoring:
            raise ValueError("Formal training cannot enable dense_scoring.")
        if not aligned_cache:
            raise ValueError("Formal training must enable aligned_cache.")
        if torch.device(device) != torch.device(contract.EXECUTION_DEVICE):
            raise ValueError(
                f"Formal training device must be {contract.EXECUTION_DEVICE}."
            )

    kwargs = {
        "output_dir": output_dir,
        "device": device,
        "resume": resume,
        "stop_after_windows": stop_after_windows,
        "aligned_cache": aligned_cache,
    }
    if dense_scoring:
        with dense_scoring_mode():
            return _run_training_impl(config, runtime_signature, **kwargs)
    return _run_training_impl(config, runtime_signature, **kwargs)


__all__ = [
    "AnchorStepResult",
    "OptimizerStepResult",
    "PreparedEffectiveGroup",
    "PreparedFilterGroup",
    "RawBatchAudit",
    "WindowCollection",
    "WindowLRScheduler",
    "WindowTrainingResult",
    "build_optimizer",
    "collect_effective_window",
    "configure_deterministic_runtime",
    "learning_rate_for_window",
    "load_groups_and_trie",
    "partition_effective_groups",
    "run_training",
    "set_global_seed",
    "train_complete_window",
    "train_minibatch",
]
