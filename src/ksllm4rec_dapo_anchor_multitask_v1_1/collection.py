"""One-pass source-block collection under one frozen policy snapshot."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch

from . import contract
from ._infra.dapo_rollout import (
    CanonicalCandidate,
    PromptRequest,
    canonicalize_prompt_rollout,
    rollout_prompt_batch,
)
from ._infra.grpo_constraint import RecommendationGrammar
from ._infra.grpo_prompt import encode_prompt
from ._infra.orpo_data import Sid
from ._infra.rloo_modeling import PolicyModel
from .data import SidTargetGroup, SidTask
from .objective import RewardOutput, group_relative_rewards_and_advantages
from .reward import GroupReward, ObjectiveRoute, RewardTier, route_group
from .source_blocks import SourceBlock


@dataclass(frozen=True)
class PreparedGroup:
    """One sampled group retained by at least one objective branch."""

    source_block: int
    epoch_index: int
    epoch_block_index: int
    source_index: int
    group: SidTargetGroup
    prompt_ids: tuple[int, ...]
    positives: tuple[Sid, ...]
    candidates: tuple[CanonicalCandidate, ...]
    reward: RewardOutput
    route: ObjectiveRoute
    proposal_target_max_logp_difference: float
    corrected_candidate_count: int
    target_forward_calls: int

    @property
    def identity(self) -> tuple[int, str, str]:
        return self.epoch_index, self.group.task.value, self.group.group_id


@dataclass(frozen=True)
class SourceGroupAudit:
    block_index: int
    epoch_index: int
    epoch_block_index: int
    task: SidTask
    source_index: int
    group_id: str
    rollout_count: int
    route: ObjectiveRoute
    policy_step_at_sample: int


@dataclass(frozen=True)
class BlockCollection:
    block_index: int
    epoch_index: int
    epoch_block_index: int
    policy_step_at_sample: int
    rl_groups: tuple[Any, ...]
    anchor_groups: tuple[Any, ...]
    source_audits: tuple[SourceGroupAudit, ...]


@dataclass(frozen=True)
class PendingOptimizationBuffer:
    snapshot_policy_step: int
    first_source_block: int | None
    next_source_block: int
    rl_groups: tuple[Any, ...]
    anchor_groups: tuple[Any, ...]
    source_audits: tuple[SourceGroupAudit, ...]

    @classmethod
    def empty(
        cls, snapshot_policy_step: int, next_source_block: int
    ) -> "PendingOptimizationBuffer":
        if snapshot_policy_step < 0 or next_source_block < 0:
            raise ValueError("Pending buffer counters must be non-negative.")
        return cls(
            snapshot_policy_step=int(snapshot_policy_step),
            first_source_block=None,
            next_source_block=int(next_source_block),
            rl_groups=(),
            anchor_groups=(),
            source_audits=(),
        )

    def append(self, block: BlockCollection) -> "PendingOptimizationBuffer":
        if block.block_index != self.next_source_block:
            raise ValueError("Source blocks must be appended contiguously.")
        if block.policy_step_at_sample != self.snapshot_policy_step:
            raise RuntimeError("A pending buffer cannot mix policy snapshots.")
        first = self.first_source_block
        if first is None:
            first = block.block_index
        rl = self.rl_groups + tuple(block.rl_groups)
        anchor = self.anchor_groups + tuple(block.anchor_groups)
        for label, values in (("RL", rl), ("Anchor", anchor)):
            identities = [_prepared_identity(item) for item in values]
            if len(identities) != len(set(identities)):
                raise RuntimeError(f"{label} pending buffer contains duplicate groups.")
        audits = self.source_audits + tuple(block.source_audits)
        audit_ids = [
            (item.epoch_index, item.task.value, item.source_index)
            for item in audits
        ]
        if len(audit_ids) != len(set(audit_ids)):
            raise RuntimeError("A source group was sampled more than once.")
        return PendingOptimizationBuffer(
            snapshot_policy_step=self.snapshot_policy_step,
            first_source_block=first,
            next_source_block=block.block_index + 1,
            rl_groups=rl,
            anchor_groups=anchor,
            source_audits=audits,
        )


def _prepared_identity(item: Any) -> tuple[Any, ...]:
    identity = getattr(item, "identity", None)
    if identity is not None:
        return tuple(identity)
    task = item.group.task.value if hasattr(item.group.task, "value") else str(item.group.task)
    epoch_index = int(getattr(item, "epoch_index", 0))
    return epoch_index, task, str(item.group.group_id)


@dataclass(frozen=True)
class CollectionResult:
    pending: PendingOptimizationBuffer
    ready_for_policy_update: bool
    source_exhausted: bool


def collect_until_policy_update(
    pending: PendingOptimizationBuffer,
    *,
    total_source_blocks: int,
    minimum_effective_groups: int,
    sample_block: Callable[[int, int], BlockCollection],
) -> CollectionResult:
    """Sample complete source blocks without allowing an optimizer callback."""

    if total_source_blocks <= 0 or minimum_effective_groups <= 0:
        raise ValueError("Collection limits must be positive.")
    current = pending
    while (
        len(current.rl_groups) < minimum_effective_groups
        and current.next_source_block < total_source_blocks
    ):
        block = sample_block(
            current.next_source_block, current.snapshot_policy_step
        )
        current = current.append(block)
    exhausted = current.next_source_block == total_source_blocks
    ready = len(current.rl_groups) >= minimum_effective_groups or (
        exhausted and bool(current.rl_groups)
    )
    return CollectionResult(
        pending=current,
        ready_for_policy_update=ready,
        source_exhausted=exhausted,
    )


def _reward_route(reward: RewardOutput) -> ObjectiveRoute:
    grouped = GroupReward(
        rewards=tuple(float(value) for value in reward.rewards.tolist()),
        tiers=tuple(RewardTier(value) for value in reward.tiers),
        effective=reward.effective,
        has_exact="exact" in reward.tiers,
    )
    return route_group(grouped).route


def _prepare_prompt(
    bundle: PolicyModel, group: SidTargetGroup, cutoff_len: int
) -> tuple[int, ...]:
    return tuple(
        encode_prompt(
            bundle.tokenizer,
            group.system,
            group.prompt,
            cutoff_len=int(cutoff_len),
        )
    )


def _sample_task_groups(
    bundle: PolicyModel,
    groups: Sequence[SidTargetGroup],
    source_indices: Sequence[int],
    grammar: RecommendationGrammar,
    *,
    block_index: int,
    epoch_index: int,
    epoch_block_index: int,
    policy_step: int,
    config: Mapping[str, Any],
    device: str | torch.device,
    aligned_cache: bool,
) -> tuple[tuple[PreparedGroup, ...], tuple[SourceGroupAudit, ...]]:
    values = tuple(groups)
    indices = tuple(int(value) for value in source_indices)
    if len(values) != len(indices):
        raise ValueError("Source groups and indices must have the same length.")
    if not values:
        return (), ()
    tasks = {group.task for group in values}
    if len(tasks) != 1:
        raise ValueError("A physical rollout batch stream must contain one task.")
    task = values[0].task
    prepared: list[PreparedGroup] = []
    audits: list[SourceGroupAudit] = []
    batch_size = int(config["rollout"]["prompt_batch_size"])
    seed = int(config["train"]["seed"])
    for start in range(0, len(values), batch_size):
        batch_groups = values[start : start + batch_size]
        batch_indices = indices[start : start + batch_size]
        requests = tuple(
            PromptRequest(
                group_id=f"{seed}|{task.value}|{group.group_id}",
                prompt_ids=_prepare_prompt(
                    bundle, group, int(config["data"]["cutoff_len"])
                ),
            )
            for group in batch_groups
        )
        proposals, _ = rollout_prompt_batch(
            bundle.model,
            requests,
            grammar,
            sampling_nonce=int(block_index),
            config=dict(config),
            device=device,
            aligned_cache=aligned_cache,
        )
        for group, source_index, proposal in zip(
            batch_groups, batch_indices, proposals, strict=True
        ):
            canonical = canonicalize_prompt_rollout(
                bundle.model,
                proposal,
                grammar,
                sampling_nonce=int(block_index),
                temperature=float(config["rollout"]["temperature"]),
                max_difference=float(
                    config["rollout"]["sample_canonical_max_logp_difference"]
                ),
                device=device,
            )
            positives = group.positives
            reward = group_relative_rewards_and_advantages(
                [candidate.sid for candidate in canonical.candidates],
                positives,
                reward_values=config["reward"],
            )
            route = _reward_route(reward)
            item = PreparedGroup(
                source_block=int(block_index),
                epoch_index=int(epoch_index),
                epoch_block_index=int(epoch_block_index),
                source_index=int(source_index),
                group=group,
                prompt_ids=proposal.request.prompt_ids,
                positives=positives,
                candidates=canonical.candidates,
                reward=reward,
                route=route,
                proposal_target_max_logp_difference=(
                    canonical.max_proposal_canonical_logp_difference
                ),
                corrected_candidate_count=canonical.corrected_candidate_count,
                target_forward_calls=canonical.target_forward_calls,
            )
            if route is ObjectiveRoute.RL_AND_ANCHOR:
                prepared.append(item)
            audits.append(
                SourceGroupAudit(
                    block_index=int(block_index),
                    epoch_index=int(epoch_index),
                    epoch_block_index=int(epoch_block_index),
                    task=task,
                    source_index=int(source_index),
                    group_id=group.group_id,
                    rollout_count=len(canonical.candidates),
                    route=route,
                    policy_step_at_sample=int(policy_step),
                )
            )
            if route is ObjectiveRoute.ANCHOR:
                prepared.append(item)
    return tuple(prepared), tuple(audits)


def sample_source_block(
    bundle: PolicyModel,
    recommendation_groups: Sequence[SidTargetGroup],
    text_to_sid_groups: Sequence[SidTargetGroup],
    source_block: SourceBlock,
    grammars: Mapping[SidTask, RecommendationGrammar],
    *,
    recommendation_order: Sequence[int] | None = None,
    text_to_sid_order: Sequence[int] | None = None,
    policy_step: int,
    config: Mapping[str, Any],
    device: str | torch.device,
    aligned_cache: bool = True,
) -> BlockCollection:
    """Sample recommendation first, then text, with no optimizer access."""

    active = config["experiments"]["active"]
    task_inputs: list[tuple[SidTask, tuple[SidTargetGroup, ...], tuple[int, ...]]] = []
    recommendation_indices = (
        tuple(range(len(recommendation_groups)))
        if recommendation_order is None
        else tuple(int(value) for value in recommendation_order)
    )
    text_indices = (
        tuple(range(len(text_to_sid_groups)))
        if text_to_sid_order is None
        else tuple(int(value) for value in text_to_sid_order)
    )
    if len(recommendation_indices) != len(recommendation_groups):
        raise ValueError("Recommendation epoch order has the wrong length.")
    if len(text_indices) != len(text_to_sid_groups):
        raise ValueError("Text-to-SID epoch order has the wrong length.")
    if bool(active["recommendation_enabled"]):
        task_inputs.append(
            (
                SidTask.RECOMMENDATION,
                tuple(
                    recommendation_groups[index]
                    for index in source_block.recommendation.select(
                        recommendation_indices
                    )
                ),
                source_block.recommendation.select(recommendation_indices),
            )
        )
    if bool(active["text_to_sid_enabled"]):
        task_inputs.append(
            (
                SidTask.ITEM_TEXT_TO_SID,
                tuple(
                    text_to_sid_groups[index]
                    for index in source_block.text_to_sid.select(text_indices)
                ),
                source_block.text_to_sid.select(text_indices),
            )
        )
    rl: list[PreparedGroup] = []
    anchor: list[PreparedGroup] = []
    audits: list[SourceGroupAudit] = []
    for task, groups, indices in task_inputs:
        sampled, task_audits = _sample_task_groups(
            bundle,
            groups,
            tuple(indices),
            grammars[task],
            block_index=source_block.index,
            epoch_index=source_block.epoch_index,
            epoch_block_index=source_block.epoch_block_index,
            policy_step=policy_step,
            config=config,
            device=device,
            aligned_cache=aligned_cache,
        )
        by_identity = {item.identity: item for item in sampled}
        for audit in task_audits:
            identity = (audit.epoch_index, audit.task.value, audit.group_id)
            item = by_identity.get(identity)
            if audit.route is ObjectiveRoute.RL_AND_ANCHOR:
                if item is None:
                    raise RuntimeError("RL route lost its prepared rollout group.")
                rl.append(item)
            if audit.route in {ObjectiveRoute.ANCHOR, ObjectiveRoute.RL_AND_ANCHOR}:
                if item is None:
                    raise RuntimeError("Anchor route lost its prepared rollout group.")
                anchor.append(item)
        audits.extend(task_audits)
    return BlockCollection(
        block_index=source_block.index,
        epoch_index=source_block.epoch_index,
        epoch_block_index=source_block.epoch_block_index,
        policy_step_at_sample=int(policy_step),
        rl_groups=tuple(rl),
        anchor_groups=tuple(anchor),
        source_audits=tuple(audits),
    )


__all__ = [
    "BlockCollection",
    "CollectionResult",
    "PendingOptimizationBuffer",
    "PreparedGroup",
    "SourceGroupAudit",
    "collect_until_policy_update",
    "sample_source_block",
]
