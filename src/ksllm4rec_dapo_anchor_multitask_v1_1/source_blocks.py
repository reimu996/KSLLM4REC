"""Deterministic source allocation for two exact multitask epochs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from typing import Sequence, TypeVar


SOURCE_BLOCKS = 532
SOURCE_EPOCHS = 2
RECOMMENDATION_GROUPS = 17_016
TEXT_TO_SID_GROUPS = 10_597
GROUP_SIZE = 16
SOURCE_GROUPS_PER_EPOCH = RECOMMENDATION_GROUPS + TEXT_TO_SID_GROUPS
TOTAL_SOURCE_GROUPS = SOURCE_GROUPS_PER_EPOCH * SOURCE_EPOCHS
TOTAL_SOURCE_BLOCKS = SOURCE_BLOCKS * SOURCE_EPOCHS
TOTAL_CANDIDATE_ROLLOUTS = TOTAL_SOURCE_GROUPS * GROUP_SIZE

T = TypeVar("T")


@dataclass(frozen=True)
class GroupRange:
    """A contiguous slice of one epoch's deterministic task ordering."""

    start: int
    stop: int

    @property
    def count(self) -> int:
        return self.stop - self.start

    def select(self, groups: Sequence[T]) -> tuple[T, ...]:
        if self.stop > len(groups):
            raise ValueError("Group range exceeds the supplied source sequence.")
        return tuple(groups[self.start : self.stop])


@dataclass(frozen=True)
class SourceBlock:
    """One global block with its epoch-local source ranges."""

    index: int
    epoch_index: int
    epoch_block_index: int
    recommendation: GroupRange
    text_to_sid: GroupRange

    @property
    def source_groups(self) -> int:
        return self.recommendation.count + self.text_to_sid.count


@dataclass(frozen=True)
class EpochSourcePlan:
    """A complete, non-repeating ordering for one semantic source epoch."""

    epoch_index: int
    recommendation_order: tuple[int, ...]
    text_to_sid_order: tuple[int, ...]
    blocks: tuple[SourceBlock, ...]
    sha256: str

    @property
    def source_groups(self) -> int:
        return len(self.recommendation_order) + len(self.text_to_sid_order)


def _range_for_block(index: int, total: int, blocks: int) -> GroupRange:
    return GroupRange(
        start=(index * total) // blocks,
        stop=((index + 1) * total) // blocks,
    )


def build_source_blocks(
    *,
    recommendation_groups: int = RECOMMENDATION_GROUPS,
    text_to_sid_groups: int = TEXT_TO_SID_GROUPS,
    blocks: int = SOURCE_BLOCKS,
    epoch_index: int = 0,
    global_offset: int = 0,
) -> tuple[SourceBlock, ...]:
    """Partition one already-ordered epoch without omission or overlap."""

    if recommendation_groups <= 0 or text_to_sid_groups <= 0 or blocks <= 0:
        raise ValueError("Source counts and block count must be positive.")
    if epoch_index < 0 or global_offset < 0:
        raise ValueError("Epoch and global block offsets must be non-negative.")
    if recommendation_groups < blocks or text_to_sid_groups < blocks:
        raise ValueError("Every source block must contain both tasks.")
    result = tuple(
        SourceBlock(
            index=global_offset + block_index,
            epoch_index=epoch_index,
            epoch_block_index=block_index,
            recommendation=_range_for_block(block_index, recommendation_groups, blocks),
            text_to_sid=_range_for_block(block_index, text_to_sid_groups, blocks),
        )
        for block_index in range(blocks)
    )
    if result[0].recommendation.start != 0 or result[0].text_to_sid.start != 0:
        raise RuntimeError("Source block plan does not begin at zero.")
    if (
        result[-1].recommendation.stop != recommendation_groups
        or result[-1].text_to_sid.stop != text_to_sid_groups
    ):
        raise RuntimeError("Source block plan does not consume both sources.")
    for previous, current in zip(result, result[1:]):
        if previous.recommendation.stop != current.recommendation.start:
            raise RuntimeError("Recommendation block ranges are not contiguous.")
        if previous.text_to_sid.stop != current.text_to_sid.start:
            raise RuntimeError("Text-to-SID block ranges are not contiguous.")
    return result


def _epoch_order(
    group_ids: Sequence[str], *, seed: int, epoch_index: int, task: str
) -> tuple[int, ...]:
    values = tuple(str(value) for value in group_ids)
    if not values or len(values) != len(set(values)):
        raise ValueError("Each task needs non-empty, unique group IDs.")

    def key(index: int) -> tuple[str, str]:
        group_id = values[index]
        payload = f"dapo-anchor-v1.1|{seed}|{epoch_index}|{task}|{group_id}"
        return sha256(payload.encode("utf-8")).hexdigest(), group_id

    return tuple(sorted(range(len(values)), key=key))


def _plan_digest(
    *,
    epoch_index: int,
    recommendation_ids: Sequence[str],
    text_ids: Sequence[str],
    recommendation_order: Sequence[int],
    text_to_sid_order: Sequence[int],
) -> str:
    payload = {
        "schema_version": 1,
        "epoch_index": int(epoch_index),
        "recommendation_group_ids": [
            str(recommendation_ids[index]) for index in recommendation_order
        ],
        "text_to_sid_group_ids": [str(text_ids[index]) for index in text_to_sid_order],
    }
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return sha256(encoded).hexdigest()


def build_epoch_source_plans(
    recommendation_group_ids: Sequence[str],
    text_to_sid_group_ids: Sequence[str],
    *,
    seed: int,
    source_epochs: int = SOURCE_EPOCHS,
    blocks: int = SOURCE_BLOCKS,
) -> tuple[EpochSourcePlan, ...]:
    """Build independent deterministic orderings for every requested epoch."""

    if source_epochs != SOURCE_EPOCHS:
        raise ValueError(f"This profile requires exactly {SOURCE_EPOCHS} source epochs.")
    recommendation_ids = tuple(str(value) for value in recommendation_group_ids)
    text_ids = tuple(str(value) for value in text_to_sid_group_ids)
    if len(recommendation_ids) != RECOMMENDATION_GROUPS:
        raise ValueError("Recommendation group count differs from the frozen contract.")
    if len(text_ids) != TEXT_TO_SID_GROUPS:
        raise ValueError("Text-to-SID group count differs from the frozen contract.")
    plans: list[EpochSourcePlan] = []
    for epoch_index in range(source_epochs):
        recommendation_order = _epoch_order(
            recommendation_ids,
            seed=seed,
            epoch_index=epoch_index,
            task="recommendation",
        )
        text_to_sid_order = _epoch_order(
            text_ids,
            seed=seed,
            epoch_index=epoch_index,
            task="item_text_to_sid",
        )
        plan_blocks = build_source_blocks(
            recommendation_groups=len(recommendation_ids),
            text_to_sid_groups=len(text_ids),
            blocks=blocks,
            epoch_index=epoch_index,
            global_offset=epoch_index * blocks,
        )
        plans.append(
            EpochSourcePlan(
                epoch_index=epoch_index,
                recommendation_order=recommendation_order,
                text_to_sid_order=text_to_sid_order,
                blocks=plan_blocks,
                sha256=_plan_digest(
                    epoch_index=epoch_index,
                    recommendation_ids=recommendation_ids,
                    text_ids=text_ids,
                    recommendation_order=recommendation_order,
                    text_to_sid_order=text_to_sid_order,
                ),
            )
        )
    return tuple(plans)


def flatten_epoch_source_blocks(
    plans: Sequence[EpochSourcePlan],
) -> tuple[SourceBlock, ...]:
    """Return the globally indexed blocks in exact epoch order."""

    values = tuple(plan for plan in plans)
    if len(values) != SOURCE_EPOCHS:
        raise ValueError(f"Expected exactly {SOURCE_EPOCHS} epoch plans.")
    flattened = tuple(block for plan in values for block in plan.blocks)
    if tuple(block.index for block in flattened) != tuple(range(len(flattened))):
        raise RuntimeError("Global source blocks are not contiguous.")
    if len(flattened) != TOTAL_SOURCE_BLOCKS:
        raise RuntimeError("Global source block count differs from the frozen contract.")
    return flattened


def source_plan_manifest(
    plans: Sequence[EpochSourcePlan],
    recommendation_group_ids: Sequence[str],
    text_to_sid_group_ids: Sequence[str],
    *,
    seed: int,
) -> dict[str, object]:
    """Return a self-verifying, human-readable frozen two-epoch plan."""

    values = tuple(plans)
    if len(values) != SOURCE_EPOCHS:
        raise ValueError(f"Expected exactly {SOURCE_EPOCHS} epoch plans.")
    recommendation_ids = tuple(str(value) for value in recommendation_group_ids)
    text_ids = tuple(str(value) for value in text_to_sid_group_ids)
    epochs = [
        {
            "epoch_index": plan.epoch_index,
            "sha256": plan.sha256,
            "recommendation_source_indices": list(plan.recommendation_order),
            "recommendation_group_ids": [
                recommendation_ids[index] for index in plan.recommendation_order
            ],
            "text_to_sid_source_indices": list(plan.text_to_sid_order),
            "text_to_sid_group_ids": [
                text_ids[index] for index in plan.text_to_sid_order
            ],
        }
        for plan in values
    ]
    payload = {
        "schema_version": 1,
        "seed": int(seed),
        "source_epochs": SOURCE_EPOCHS,
        "source_blocks_per_epoch": SOURCE_BLOCKS,
        "epochs": epochs,
    }
    canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode(
        "utf-8"
    )
    return {**payload, "sha256": sha256(canonical).hexdigest()}


def candidate_rollout_count(
    recommendation_groups: int,
    text_to_sid_groups: int,
    *,
    group_size: int = GROUP_SIZE,
    source_epochs: int = 1,
) -> int:
    if (
        recommendation_groups < 0
        or text_to_sid_groups < 0
        or group_size <= 0
        or source_epochs <= 0
    ):
        raise ValueError("Source counts, group size, and epochs must be positive.")
    return (recommendation_groups + text_to_sid_groups) * group_size * source_epochs


__all__ = [
    "EpochSourcePlan",
    "GROUP_SIZE",
    "RECOMMENDATION_GROUPS",
    "SOURCE_BLOCKS",
    "SOURCE_EPOCHS",
    "SOURCE_GROUPS_PER_EPOCH",
    "TEXT_TO_SID_GROUPS",
    "TOTAL_CANDIDATE_ROLLOUTS",
    "TOTAL_SOURCE_BLOCKS",
    "TOTAL_SOURCE_GROUPS",
    "GroupRange",
    "SourceBlock",
    "build_epoch_source_plans",
    "build_source_blocks",
    "candidate_rollout_count",
    "flatten_epoch_source_blocks",
    "source_plan_manifest",
]
