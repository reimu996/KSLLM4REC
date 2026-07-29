"""Mixed-task RL backward and GT-set Anchor gradient construction."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping, Sequence

import torch

from . import contract
from ._infra.dapo_rollout import CanonicalCandidate
from ._infra.dapo_scoring import (
    grammar_completion_width,
    score_completions_dense_fixed,
)
from ._infra.grpo_constraint import RecommendationGrammar
from ._infra.rloo_modeling import PolicyModel
from .anchor_scoring import score_prompt_completion_pairs_dense
from .collection import PendingOptimizationBuffer, PreparedGroup
from .data import SidTask
from .engine import (
    FinalAuxiliaryExecution,
    PolicyStepLRScheduler,
    PolicyUpdateExecution,
    execute_policy_updates,
    execute_final_auxiliary_flush,
    gradient_l2_norm,
    partition_policy_groups,
)
from .objective import (
    clipped_group_relative_token_sum,
    gt_sequence_logps,
    gt_set_anchor_loss,
)


@dataclass(frozen=True)
class PolicyBackwardResult:
    loss: float
    learning_rate: float
    decision_tokens: int
    max_replay_logp_difference: float
    ratio_min: float
    ratio_max: float
    ratio_mean: float
    clipped_tokens: int
    below_low_tokens: int
    above_high_tokens: int
    group_ids: tuple[str, ...]
    tasks: tuple[str, ...]
    epoch_indexes: tuple[int, ...]


@dataclass(frozen=True)
class AnchorGroupResult:
    epoch_index: int
    source_block: int
    epoch_block_index: int
    task: str
    group_id: str
    set_nll: float
    all_gt_decision_nll_sum: float
    gt_sid_count: int
    decision_tokens: int


@dataclass(frozen=True)
class AnchorRawGradientResult:
    group_results: tuple[AnchorGroupResult, ...]
    raw_gradient_norm: float
    max_replay_logp_difference: float


@dataclass(frozen=True)
class PolicyWindowResult:
    optimization_window_index: int
    first_source_block: int
    next_source_block: int
    policy_optimizer_steps: int
    anchor_optimizer_steps: int
    group_ids: tuple[str, ...]
    tasks: tuple[str, ...]
    epoch_indexes: tuple[int, ...]
    execution: PolicyUpdateExecution


def trainable_parameters(bundle: PolicyModel) -> list[torch.nn.Parameter]:
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
    return torch.optim.AdamW(trainable_parameters(bundle), **kwargs)


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


def backward_policy_minibatch(
    bundle: PolicyModel,
    groups: Sequence[PreparedGroup],
    grammars: Mapping[SidTask, RecommendationGrammar],
    *,
    config: Mapping[str, Any],
    device: str | torch.device,
    enforce_replay_gate: bool,
) -> PolicyBackwardResult:
    """Backpropagate one 1..8-group RL minibatch without stepping AdamW."""

    values = tuple(groups)
    if not 1 <= len(values) <= contract.MINIBATCH_GROUPS:
        raise ValueError("A policy minibatch must contain one to eight groups.")
    torch_device = torch.device(device)
    total_decisions = sum(
        sum(candidate.decision_mask)
        for group in values
        for candidate in group.candidates
    )
    if total_decisions <= 0:
        raise RuntimeError("Policy minibatch contains no decision token.")
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
        advantages = prepared.reward.advantages.to(torch_device)
        for start in range(0, contract.GROUP_SIZE, contract.LOSS_CHUNK_SIZE):
            chunk = prepared.candidates[start : start + contract.LOSS_CHUNK_SIZE]
            scores = score_completions_dense_fixed(
                bundle.model,
                prepared.prompt_ids,
                [candidate.token_ids for candidate in chunk],
                grammar,
                device=torch_device,
                temperature=float(config["rollout"]["temperature"]),
                batch_rows=len(chunk),
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
                advantages[start : start + len(chunk)],
                scores.decision_mask,
                clip_low=float(config["loss"]["clip_ratio_low"]),
                clip_high=float(config["loss"]["clip_ratio_high"]),
            )
            (output.loss_sum / float(total_decisions)).backward()
            loss_sum += float(output.loss_sum.detach().item())
            ratios.extend(float(value) for value in output.ratios.cpu().tolist())
            clipped_tokens += int(output.clip_active.sum().item())
            below_tokens += int(output.below_low.sum().item())
            above_tokens += int(output.above_high.sum().item())
    if not ratios or not all(math.isfinite(value) and value > 0.0 for value in ratios):
        raise FloatingPointError("Policy ratios are not positive finite values.")
    return PolicyBackwardResult(
        loss=loss_sum / float(total_decisions),
        learning_rate=0.0,
        decision_tokens=total_decisions,
        max_replay_logp_difference=maximum_delta,
        ratio_min=min(ratios),
        ratio_max=max(ratios),
        ratio_mean=sum(ratios) / len(ratios),
        clipped_tokens=clipped_tokens,
        below_low_tokens=below_tokens,
        above_high_tokens=above_tokens,
        group_ids=tuple(item.group.group_id for item in values),
        tasks=tuple(item.group.task.value for item in values),
        epoch_indexes=tuple(int(item.epoch_index) for item in values),
    )


def _encode_anchor_completions(
    group: PreparedGroup, grammar: RecommendationGrammar
) -> tuple[tuple[int, ...], ...]:
    if not group.positives:
        raise RuntimeError("An Anchor group has no GT SID.")
    completions = tuple(tuple(grammar.encode_sid(sid)) for sid in group.positives)
    if len(set(completions)) != len(completions):
        raise RuntimeError("Distinct GT SIDs encoded to duplicate paths.")
    return completions


@dataclass(frozen=True)
class _AnchorRow:
    group_index: int
    gt_index: int
    task: SidTask
    group_id: str
    prompt_ids: tuple[int, ...]
    completion_ids: tuple[int, ...]


@dataclass(frozen=True)
class _AnchorReplayBatch:
    task: SidTask
    rows: tuple[_AnchorRow, ...]
    first_logps: torch.Tensor
    first_decision_mask: torch.Tensor
    first_decision_counts: torch.Tensor


def build_anchor_raw_gradient(
    bundle: PolicyModel,
    optimizer: torch.optim.Optimizer,
    anchor_groups: Sequence[PreparedGroup],
    grammars: Mapping[SidTask, RecommendationGrammar],
    *,
    config: Mapping[str, Any],
    device: str | torch.device,
) -> AnchorRawGradientResult:
    """Average every Anchor group across both tasks into one raw gradient."""

    groups = tuple(anchor_groups)
    if not groups:
        raise ValueError("Raw Anchor gradient requires at least one group.")
    identities = [item.identity for item in groups]
    if len(identities) != len(set(identities)):
        raise RuntimeError("Anchor group identities must be unique.")
    microbatch = int(config["anchor"]["gt_completion_microbatch"])
    if microbatch != contract.ANCHOR_GT_COMPLETION_MICROBATCH:
        raise ValueError("Multitask V1 fixes Anchor rows per batch at eight.")
    torch_device = torch.device(device)
    rows: list[_AnchorRow] = []
    completion_counts: list[int] = []
    for group_index, item in enumerate(groups):
        grammar = grammars[item.group.task]
        completions = _encode_anchor_completions(item, grammar)
        completion_counts.append(len(completions))
        rows.extend(
            _AnchorRow(
                group_index=group_index,
                gt_index=gt_index,
                task=item.group.task,
                group_id=item.group.group_id,
                prompt_ids=item.prompt_ids,
                completion_ids=completion,
            )
            for gt_index, completion in enumerate(completions)
        )
    sequence_by_group = [
        torch.empty(count, dtype=torch.float32) for count in completion_counts
    ]
    decision_by_group = [
        torch.empty(count, dtype=torch.long) for count in completion_counts
    ]
    replay_batches: list[_AnchorReplayBatch] = []
    optimizer.zero_grad(set_to_none=True)
    with torch.no_grad():
        for task in SidTask:
            task_rows = sorted(
                (row for row in rows if row.task is task),
                key=lambda row: (len(row.prompt_ids), row.group_id, row.gt_index),
            )
            grammar = grammars[task]
            width = grammar_completion_width(grammar)
            for start in range(0, len(task_rows), microbatch):
                batch_rows = tuple(task_rows[start : start + microbatch])
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
                    _AnchorReplayBatch(
                        task=task,
                        rows=batch_rows,
                        first_logps=scores.log_probs.detach().cpu(),
                        first_decision_mask=scores.decision_mask.detach().cpu(),
                        first_decision_counts=counts.detach().cpu(),
                    )
                )
    group_results: list[AnchorGroupResult] = []
    weights_by_group: list[torch.Tensor] = []
    for item, sequences, counts in zip(
        groups, sequence_by_group, decision_by_group, strict=True
    ):
        group_results.append(
            AnchorGroupResult(
                epoch_index=int(getattr(item, "epoch_index", 0)),
                source_block=int(getattr(item, "source_block", 0)),
                epoch_block_index=int(getattr(item, "epoch_block_index", 0)),
                task=item.group.task.value,
                group_id=item.group.group_id,
                set_nll=float(gt_set_anchor_loss(sequences).item()),
                all_gt_decision_nll_sum=float((-sequences.sum()).item()),
                gt_sid_count=sequences.numel(),
                decision_tokens=int(counts.sum().item()),
            )
        )
        weights_by_group.append(sequences.softmax(dim=0))
    maximum_delta = 0.0
    limit = float(config["anchor"]["replay_max_logp_difference"])
    for replay in replay_batches:
        grammar = grammars[replay.task]
        scores = score_prompt_completion_pairs_dense(
            bundle.model,
            [row.prompt_ids for row in replay.rows],
            [row.completion_ids for row in replay.rows],
            grammar,
            device=torch_device,
            temperature=float(config["anchor"]["teacher_forcing_temperature"]),
            completion_width=grammar_completion_width(grammar),
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
                "Anchor no-grad/grad replay gate failed: "
                f"max_abs_delta={maximum_delta:.9g}, limit={limit:.9g}."
            )
        row_weights = torch.stack(
            [
                weights_by_group[row.group_index][row.gt_index]
                for row in replay.rows
            ]
        ).to(torch_device)
        surrogate = -(row_weights * sequences).sum() / float(len(groups))
        if surrogate.requires_grad:
            surrogate.backward()
    return AnchorRawGradientResult(
        group_results=tuple(group_results),
        raw_gradient_norm=gradient_l2_norm(trainable_parameters(bundle)),
        max_replay_logp_difference=maximum_delta,
    )


def optimize_pending_buffer(
    bundle: PolicyModel,
    optimizer: torch.optim.Optimizer,
    scheduler: PolicyStepLRScheduler,
    pending: PendingOptimizationBuffer,
    grammars: Mapping[SidTask, RecommendationGrammar],
    *,
    optimization_window_index: int,
    config: Mapping[str, Any],
    device: str | torch.device,
) -> PolicyWindowResult:
    if not pending.rl_groups:
        raise ValueError("A policy window requires at least one RL group.")
    if pending.snapshot_policy_step != scheduler.completed_policy_steps:
        raise RuntimeError("Pending candidates belong to another policy snapshot.")
    batches = partition_policy_groups(
        pending.rl_groups,
        optimization_window_index=int(optimization_window_index),
        seed=int(config["train"]["seed"]),
    )
    parameters = trainable_parameters(bundle)

    def backward_policy(index: int, batch: Sequence[PreparedGroup]):
        result = backward_policy_minibatch(
            bundle,
            batch,
            grammars,
            config=config,
            device=device,
            enforce_replay_gate=index == 0,
        )
        return PolicyBackwardResult(
            **{
                **result.__dict__,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

    def backward_anchor():
        return build_anchor_raw_gradient(
            bundle,
            optimizer,
            pending.anchor_groups,
            grammars,
            config=config,
            device=device,
        )

    execution = execute_policy_updates(
        parameters,
        optimizer,
        scheduler,
        batches,
        backward_policy=backward_policy,
        backward_anchor=backward_anchor if pending.anchor_groups else None,
        max_anchor_weight=float(config["anchor"]["max_weight"]),
        target_gradient_ratio=float(config["anchor"]["target_gradient_ratio"]),
        epsilon=float(config["anchor"]["gradient_epsilon"]),
        max_grad_norm=float(config["train"]["max_grad_norm"]),
    )
    identities = [item.identity for batch in batches for item in batch]
    if len(identities) != len(pending.rl_groups) or len(set(identities)) != len(identities):
        raise RuntimeError("K=1 invariant failed after policy optimization.")
    if pending.first_source_block is None:
        raise RuntimeError("A non-empty pending buffer has no first source block.")
    return PolicyWindowResult(
        optimization_window_index=int(optimization_window_index),
        first_source_block=pending.first_source_block,
        next_source_block=pending.next_source_block,
        policy_optimizer_steps=execution.policy_optimizer_steps,
        anchor_optimizer_steps=0,
        group_ids=tuple(group_id for _, _, group_id in identities),
        tasks=tuple(task for _, task, _ in identities),
        epoch_indexes=tuple(epoch_index for epoch_index, _, _ in identities),
        execution=execution,
    )


def flush_pending_anchor(
    bundle: PolicyModel,
    optimizer: torch.optim.Optimizer,
    pending: PendingOptimizationBuffer,
    grammars: Mapping[SidTask, RecommendationGrammar],
    *,
    last_rl_reference_gradient_norm: float,
    config: Mapping[str, Any],
    device: str | torch.device,
) -> FinalAuxiliaryExecution:
    if pending.rl_groups:
        raise ValueError("Final auxiliary flush requires no pending RL groups.")
    if not pending.anchor_groups:
        raise ValueError("Final auxiliary flush requires pending Anchor groups.")
    parameters = trainable_parameters(bundle)

    def backward_anchor():
        return build_anchor_raw_gradient(
            bundle,
            optimizer,
            pending.anchor_groups,
            grammars,
            config=config,
            device=device,
        )

    return execute_final_auxiliary_flush(
        parameters,
        optimizer,
        backward_anchor=backward_anchor,
        max_anchor_weight=float(config["anchor"]["max_weight"]),
        rl_reference_gradient_norm=float(last_rl_reference_gradient_norm),
        target_gradient_ratio=float(config["anchor"]["target_gradient_ratio"]),
        epsilon=float(config["anchor"]["gradient_epsilon"]),
        max_grad_norm=float(config["train"]["max_grad_norm"]),
    )


__all__ = [
    "AnchorGroupResult",
    "AnchorRawGradientResult",
    "PolicyBackwardResult",
    "PolicyWindowResult",
    "backward_policy_minibatch",
    "build_anchor_raw_gradient",
    "build_optimizer",
    "flush_pending_anchor",
    "optimize_pending_buffer",
    "trainable_parameters",
]
