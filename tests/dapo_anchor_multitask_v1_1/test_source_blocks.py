from __future__ import annotations

import unittest

from ksllm4rec_dapo_anchor_multitask_v1_1.source_blocks import (
    GROUP_SIZE,
    RECOMMENDATION_GROUPS,
    SOURCE_BLOCKS,
    SOURCE_EPOCHS,
    SOURCE_GROUPS_PER_EPOCH,
    TEXT_TO_SID_GROUPS,
    TOTAL_CANDIDATE_ROLLOUTS,
    TOTAL_SOURCE_BLOCKS,
    TOTAL_SOURCE_GROUPS,
    build_epoch_source_plans,
    build_source_blocks,
    candidate_rollout_count,
    flatten_epoch_source_blocks,
    source_plan_manifest,
)


def _group_ids(prefix: str, count: int) -> tuple[str, ...]:
    return tuple(f"{prefix}-{index:05d}" for index in range(count))


class SourceBlockTest(unittest.TestCase):
    def test_one_epoch_has_532_contiguous_task_ranges(self) -> None:
        blocks = build_source_blocks()
        self.assertEqual(len(blocks), SOURCE_BLOCKS)
        self.assertEqual(
            [block.index for block in blocks], list(range(SOURCE_BLOCKS))
        )
        self.assertEqual({block.epoch_index for block in blocks}, {0})
        self.assertEqual(
            [block.epoch_block_index for block in blocks], list(range(SOURCE_BLOCKS))
        )
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
        self.assertEqual(sum(block.source_groups for block in blocks), SOURCE_GROUPS_PER_EPOCH)

    def test_two_epoch_plan_visits_each_normalized_group_once_per_epoch(self) -> None:
        recommendation_ids = _group_ids("rec", RECOMMENDATION_GROUPS)
        text_ids = _group_ids("text", TEXT_TO_SID_GROUPS)
        plans = build_epoch_source_plans(recommendation_ids, text_ids, seed=42)
        flattened = flatten_epoch_source_blocks(plans)

        self.assertEqual(len(plans), SOURCE_EPOCHS)
        self.assertEqual(len(flattened), TOTAL_SOURCE_BLOCKS)
        self.assertEqual(
            [block.index for block in flattened], list(range(TOTAL_SOURCE_BLOCKS))
        )
        for epoch_index, plan in enumerate(plans):
            self.assertEqual(plan.epoch_index, epoch_index)
            self.assertEqual(plan.source_groups, SOURCE_GROUPS_PER_EPOCH)
            self.assertEqual(set(plan.recommendation_order), set(range(RECOMMENDATION_GROUPS)))
            self.assertEqual(set(plan.text_to_sid_order), set(range(TEXT_TO_SID_GROUPS)))
            self.assertEqual(len(plan.blocks), SOURCE_BLOCKS)
            self.assertEqual(plan.blocks[0].index, epoch_index * SOURCE_BLOCKS)
            self.assertEqual(plan.blocks[-1].index, (epoch_index + 1) * SOURCE_BLOCKS - 1)
            self.assertEqual(
                sum(block.source_groups for block in plan.blocks),
                SOURCE_GROUPS_PER_EPOCH,
            )

        self.assertNotEqual(plans[0].recommendation_order, plans[1].recommendation_order)
        self.assertNotEqual(plans[0].text_to_sid_order, plans[1].text_to_sid_order)

    def test_plan_manifest_binds_epoch_orders_and_original_source_indices(self) -> None:
        recommendation_ids = _group_ids("rec", RECOMMENDATION_GROUPS)
        text_ids = _group_ids("text", TEXT_TO_SID_GROUPS)
        plans = build_epoch_source_plans(recommendation_ids, text_ids, seed=42)
        manifest = source_plan_manifest(plans, recommendation_ids, text_ids, seed=42)

        self.assertEqual(manifest["source_epochs"], 2)
        self.assertEqual(manifest["source_blocks_per_epoch"], 532)
        self.assertEqual(len(manifest["epochs"]), 2)
        self.assertRegex(str(manifest["sha256"]), r"^[0-9a-f]{64}$")
        for epoch_index, epoch in enumerate(manifest["epochs"]):
            self.assertEqual(epoch["epoch_index"], epoch_index)
            self.assertEqual(
                set(epoch["recommendation_source_indices"]),
                set(range(RECOMMENDATION_GROUPS)),
            )
            self.assertEqual(
                set(epoch["text_to_sid_source_indices"]),
                set(range(TEXT_TO_SID_GROUPS)),
            )
            self.assertEqual(
                epoch["recommendation_group_ids"],
                [recommendation_ids[index] for index in epoch["recommendation_source_indices"]],
            )
            self.assertEqual(
                epoch["text_to_sid_group_ids"],
                [text_ids[index] for index in epoch["text_to_sid_source_indices"]],
            )

    def test_same_seed_rebuilds_exactly_the_same_two_epoch_plan(self) -> None:
        recommendation_ids = _group_ids("rec", RECOMMENDATION_GROUPS)
        text_ids = _group_ids("text", TEXT_TO_SID_GROUPS)
        first = build_epoch_source_plans(recommendation_ids, text_ids, seed=42)
        second = build_epoch_source_plans(recommendation_ids, text_ids, seed=42)
        changed_seed = build_epoch_source_plans(recommendation_ids, text_ids, seed=43)

        self.assertEqual(first, second)
        self.assertNotEqual(
            first[0].recommendation_order, changed_seed[0].recommendation_order
        )
        self.assertNotEqual(first[0].sha256, changed_seed[0].sha256)

    def test_two_epoch_candidate_total_is_exactly_883616(self) -> None:
        self.assertEqual(SOURCE_GROUPS_PER_EPOCH, 27_613)
        self.assertEqual(TOTAL_SOURCE_GROUPS, 55_226)
        self.assertEqual(TOTAL_SOURCE_BLOCKS, 1_064)
        self.assertEqual(TOTAL_CANDIDATE_ROLLOUTS, 883_616)
        self.assertEqual(
            candidate_rollout_count(
                RECOMMENDATION_GROUPS,
                TEXT_TO_SID_GROUPS,
                source_epochs=SOURCE_EPOCHS,
            ),
            TOTAL_CANDIDATE_ROLLOUTS,
        )
        self.assertEqual(TOTAL_SOURCE_GROUPS * GROUP_SIZE, TOTAL_CANDIDATE_ROLLOUTS)


if __name__ == "__main__":
    unittest.main()
