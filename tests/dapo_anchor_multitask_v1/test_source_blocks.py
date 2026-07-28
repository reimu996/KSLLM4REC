from __future__ import annotations

import unittest

from ksllm4rec_dapo_anchor_multitask_v1.source_blocks import (
    GROUP_SIZE,
    RECOMMENDATION_GROUPS,
    SOURCE_BLOCKS,
    TEXT_TO_SID_GROUPS,
    TOTAL_CANDIDATE_ROLLOUTS,
    build_source_blocks,
    candidate_rollout_count,
)


class SourceBlockTest(unittest.TestCase):
    def test_532_blocks_cover_each_source_index_exactly_once(self) -> None:
        blocks = build_source_blocks()
        self.assertEqual(len(blocks), SOURCE_BLOCKS)
        recommendation_indices = [
            index
            for block in blocks
            for index in range(block.recommendation.start, block.recommendation.stop)
        ]
        text_indices = [
            index
            for block in blocks
            for index in range(block.text_to_sid.start, block.text_to_sid.stop)
        ]
        self.assertEqual(recommendation_indices, list(range(RECOMMENDATION_GROUPS)))
        self.assertEqual(text_indices, list(range(TEXT_TO_SID_GROUPS)))
        self.assertEqual(
            {block.recommendation.count for block in blocks}, {31, 32}
        )
        self.assertEqual({block.text_to_sid.count for block in blocks}, {19, 20})

    def test_candidate_total_is_exactly_441808(self) -> None:
        self.assertEqual(TOTAL_CANDIDATE_ROLLOUTS, 441_808)
        self.assertEqual(
            candidate_rollout_count(RECOMMENDATION_GROUPS, TEXT_TO_SID_GROUPS),
            TOTAL_CANDIDATE_ROLLOUTS,
        )
        self.assertEqual(
            sum(block.source_groups * GROUP_SIZE for block in build_source_blocks()),
            TOTAL_CANDIDATE_ROLLOUTS,
        )


if __name__ == "__main__":
    unittest.main()
