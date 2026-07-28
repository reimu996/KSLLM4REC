"""One-pass source allocation for the multitask DAPO-Anchor profile."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar


SOURCE_BLOCKS = 532
RECOMMENDATION_GROUPS = 17_016
TEXT_TO_SID_GROUPS = 10_597
GROUP_SIZE = 16
TOTAL_CANDIDATE_ROLLOUTS = (
    RECOMMENDATION_GROUPS + TEXT_TO_SID_GROUPS
) * GROUP_SIZE

T = TypeVar("T")


@dataclass(frozen=True)
class GroupRange:
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
    index: int
    recommendation: GroupRange
    text_to_sid: GroupRange

    @property
    def source_groups(self) -> int:
        return self.recommendation.count + self.text_to_sid.count


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
) -> tuple[SourceBlock, ...]:
    if recommendation_groups <= 0 or text_to_sid_groups <= 0 or blocks <= 0:
        raise ValueError("Source counts and block count must be positive.")
    if recommendation_groups < blocks or text_to_sid_groups < blocks:
        raise ValueError("Every source block must contain both tasks.")
    result = tuple(
        SourceBlock(
            index=index,
            recommendation=_range_for_block(index, recommendation_groups, blocks),
            text_to_sid=_range_for_block(index, text_to_sid_groups, blocks),
        )
        for index in range(blocks)
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


def candidate_rollout_count(
    recommendation_groups: int,
    text_to_sid_groups: int,
    *,
    group_size: int = GROUP_SIZE,
) -> int:
    if recommendation_groups < 0 or text_to_sid_groups < 0 or group_size <= 0:
        raise ValueError("Group counts must be non-negative and group_size positive.")
    return (recommendation_groups + text_to_sid_groups) * group_size
