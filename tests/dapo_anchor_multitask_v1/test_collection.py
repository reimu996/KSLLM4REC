from __future__ import annotations

from types import SimpleNamespace
import unittest

from ksllm4rec_dapo_anchor_multitask_v1.collection import (
    BlockCollection,
    PendingOptimizationBuffer,
    SourceGroupAudit,
    collect_until_policy_update,
)
from ksllm4rec_dapo_anchor_multitask_v1.data import SidTask
from ksllm4rec_dapo_anchor_multitask_v1.reward import ObjectiveRoute
from ksllm4rec_dapo_anchor_multitask_v1.trainer import _periodic_checkpoint_due


def prepared(task: SidTask, group_id: str):
    return SimpleNamespace(
        group=SimpleNamespace(task=task, group_id=group_id)
    )


class FrozenPolicyCollectionTest(unittest.TestCase):
    def test_source_exhaustion_is_checkpointed_only_after_finalization(self) -> None:
        self.assertTrue(
            _periodic_checkpoint_due(
                25, requested_stop=532, total_blocks=532, interval=25
            )
        )
        self.assertTrue(
            _periodic_checkpoint_due(
                17, requested_stop=17, total_blocks=532, interval=25
            )
        )
        self.assertFalse(
            _periodic_checkpoint_due(
                532, requested_stop=532, total_blocks=532, interval=25
            )
        )

    def test_multiple_blocks_are_sampled_before_any_policy_update(self) -> None:
        sampled_policy_steps: list[int] = []

        def sample_block(block_index: int, policy_step: int) -> BlockCollection:
            sampled_policy_steps.append(policy_step)
            task_order = (SidTask.RECOMMENDATION, SidTask.ITEM_TEXT_TO_SID)
            groups = tuple(
                prepared(task_order[index % 2], f"b{block_index}-g{index}")
                for index in range(20)
            )
            audits = tuple(
                SourceGroupAudit(
                    block_index=block_index,
                    task=item.group.task,
                    source_index=block_index * 20 + index,
                    group_id=item.group.group_id,
                    rollout_count=16,
                    route=ObjectiveRoute.RL,
                    policy_step_at_sample=policy_step,
                )
                for index, item in enumerate(groups)
            )
            return BlockCollection(
                block_index=block_index,
                policy_step_at_sample=policy_step,
                rl_groups=groups,
                anchor_groups=(),
                source_audits=audits,
            )

        pending = PendingOptimizationBuffer.empty(
            snapshot_policy_step=0, next_source_block=0
        )
        result = collect_until_policy_update(
            pending,
            total_source_blocks=3,
            minimum_effective_groups=32,
            sample_block=sample_block,
        )

        self.assertEqual(sampled_policy_steps, [0, 0])
        self.assertEqual(result.pending.first_source_block, 0)
        self.assertEqual(result.pending.next_source_block, 2)
        self.assertEqual(len(result.pending.rl_groups), 40)
        self.assertTrue(result.ready_for_policy_update)
        self.assertEqual(
            {audit.policy_step_at_sample for audit in result.pending.source_audits},
            {0},
        )

    def test_last_block_is_retained_even_when_it_crosses_32(self) -> None:
        def sample_block(block_index: int, policy_step: int) -> BlockCollection:
            count = 31 if block_index == 0 else 7
            groups = tuple(
                prepared(SidTask.RECOMMENDATION, f"b{block_index}-g{index}")
                for index in range(count)
            )
            return BlockCollection(
                block_index=block_index,
                policy_step_at_sample=policy_step,
                rl_groups=groups,
                anchor_groups=(),
                source_audits=(),
            )

        result = collect_until_policy_update(
            PendingOptimizationBuffer.empty(3, 0),
            total_source_blocks=2,
            minimum_effective_groups=32,
            sample_block=sample_block,
        )
        self.assertEqual(len(result.pending.rl_groups), 38)
        self.assertEqual(result.pending.next_source_block, 2)
        self.assertTrue(result.source_exhausted)

    def test_source_end_partial_rl_window_is_ready_without_replenishment(self) -> None:
        def sample_block(block_index: int, policy_step: int) -> BlockCollection:
            groups = tuple(
                prepared(SidTask.RECOMMENDATION, f"b{block_index}-g{index}")
                for index in range(7)
            )
            return BlockCollection(
                block_index=block_index,
                policy_step_at_sample=policy_step,
                rl_groups=groups,
                anchor_groups=(),
                source_audits=(),
            )

        result = collect_until_policy_update(
            PendingOptimizationBuffer.empty(11, 0),
            total_source_blocks=1,
            minimum_effective_groups=32,
            sample_block=sample_block,
        )

        self.assertTrue(result.source_exhausted)
        self.assertTrue(result.ready_for_policy_update)
        self.assertEqual(len(result.pending.rl_groups), 7)

    def test_source_end_anchor_only_buffer_is_not_misreported_as_rl_ready(self) -> None:
        def sample_block(block_index: int, policy_step: int) -> BlockCollection:
            anchors = tuple(
                prepared(SidTask.ITEM_TEXT_TO_SID, f"a-{index}")
                for index in range(5)
            )
            return BlockCollection(
                block_index=block_index,
                policy_step_at_sample=policy_step,
                rl_groups=(),
                anchor_groups=anchors,
                source_audits=(),
            )

        result = collect_until_policy_update(
            PendingOptimizationBuffer.empty(11, 0),
            total_source_blocks=1,
            minimum_effective_groups=32,
            sample_block=sample_block,
        )

        self.assertTrue(result.source_exhausted)
        self.assertFalse(result.ready_for_policy_update)
        self.assertEqual(len(result.pending.anchor_groups), 5)


if __name__ == "__main__":
    unittest.main()
