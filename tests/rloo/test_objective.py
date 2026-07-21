from __future__ import annotations

import math
import unittest

import torch

from ksllm4rec_orpo.data import Sid
from ksllm4rec_rloo.objective import (
    ObjectiveBranch,
    calibrate_anchor_lambda,
    gt_sequence_logps,
    gt_set_anchor_loss,
    rewards_and_advantages,
    rloo_advantages,
    rloo_chunk_loss,
    rloo_loss,
    select_objective_branch,
    window_loss,
)


GT = Sid("video", 1, 2, 3)


class RewardAndBranchTest(unittest.TestCase):
    def test_five_tiers_and_highest_reward_across_multiple_gt(self) -> None:
        second_gt = Sid("prod", 9, 8, 7)
        candidates = [
            GT,
            Sid("video", 1, 2, 99),
            Sid("video", 1, 99, 99),
            Sid("video", 99, 99, 99),
            Sid("ad", 1, 2, 3),
            second_gt,
        ] + [Sid("living", 100 + i, 200 + i, 300 + i) for i in range(10)]

        result = rewards_and_advantages(candidates, [GT, second_gt])

        torch.testing.assert_close(
            result.rewards[:6],
            torch.tensor([1.0, 0.40, 0.15, 0.01, 0.0, 1.0]),
        )
        self.assertEqual(
            result.tiers[:6],
            ("exact", "same_ab", "same_a", "same_domain", "other_domain", "exact"),
        )
        self.assertEqual(result.branch, ObjectiveBranch.RLOO)

    def test_rloo_matches_leave_one_out_formula_without_std(self) -> None:
        rewards = torch.tensor(
            [1.0, 0.40, 0.15, 0.01] + [0.0] * 12, dtype=torch.float32
        )
        advantages = rloo_advantages(rewards)
        expected = (16.0 / 15.0) * (rewards - rewards.mean())
        torch.testing.assert_close(advantages, expected)
        self.assertAlmostEqual(float(advantages.sum()), 0.0, places=6)

    def test_equal_non_exact_uses_anchor_and_all_exact_skips(self) -> None:
        same_domain = torch.full((16,), 0.01)
        all_exact = torch.ones(16)
        self.assertEqual(
            select_objective_branch(same_domain), ObjectiveBranch.GT_SET_ANCHOR
        )
        self.assertEqual(select_objective_branch(all_exact), ObjectiveBranch.SKIP)
        torch.testing.assert_close(rloo_advantages(same_domain), torch.zeros(16))

    def test_one_exact_and_fifteen_equal_candidates_still_use_rloo(self) -> None:
        rewards = torch.tensor([1.0] + [0.01] * 15)
        self.assertEqual(select_objective_branch(rewards), ObjectiveBranch.RLOO)

    def test_group_must_have_exactly_sixteen_live_candidates(self) -> None:
        candidates = [Sid("ad", i, i, i) for i in range(15)]
        with self.assertRaisesRegex(ValueError, "exactly 16"):
            rewards_and_advantages(candidates, [GT])


class RlooLossTest(unittest.TestCase):
    def test_only_decision_tokens_contribute_with_one_group_denominator(self) -> None:
        logps = torch.full((16, 3), float("nan"), dtype=torch.float32)
        logps[0, :2] = torch.tensor([-0.2, -0.3])
        logps[1, 0] = -0.4
        logps.requires_grad_()
        mask = torch.zeros((16, 3), dtype=torch.bool)
        mask[0, :2] = True
        mask[1, 0] = True
        advantages = torch.zeros(16)
        advantages[0] = 1.0
        advantages[1] = -1.0

        result = rloo_loss(logps, advantages, mask)

        self.assertAlmostEqual(result.loss.item(), 0.1 / 3.0, places=6)
        self.assertEqual(result.decision_counts.tolist()[:3], [2, 1, 0])
        result.loss.backward()
        self.assertAlmostEqual(logps.grad[0, 0].item(), -1.0 / 3.0, places=6)
        self.assertAlmostEqual(logps.grad[1, 0].item(), 1.0 / 3.0, places=6)
        self.assertTrue(torch.isnan(logps.grad[2:, :]).logical_not().all())

    def test_non_rloo_advantages_are_rejected(self) -> None:
        logps = torch.zeros((16, 1), requires_grad=True)
        mask = torch.ones((16, 1), dtype=torch.bool)
        with self.assertRaisesRegex(ValueError, "sum to zero"):
            rloo_loss(logps, torch.ones(16), mask)

    def test_two_chunks_equal_the_full_group_loss_and_gradient(self) -> None:
        torch.manual_seed(7)
        base = torch.randn(16, 5, dtype=torch.float32)
        mask = torch.tensor(
            [[True, False, True, True, False]] * 16, dtype=torch.bool
        )
        rewards = torch.tensor(
            [1.0, 0.4, 0.15, 0.01, 0.0, 0.4, 0.15, 0.01] * 2
        )
        advantages = rloo_advantages(rewards)

        full_values = base.clone().requires_grad_(True)
        full = rloo_loss(full_values, advantages, mask).loss
        full.backward()

        chunk_values = base.clone().requires_grad_(True)
        denominator = int(mask.sum().item())
        chunks = []
        for start in (0, 8):
            chunks.append(
                rloo_chunk_loss(
                    chunk_values[start : start + 8],
                    advantages[start : start + 8],
                    mask[start : start + 8],
                    total_group_decisions=denominator,
                ).loss
            )
        chunk_total = sum(chunks)
        chunk_total.backward()

        torch.testing.assert_close(full, chunk_total)
        torch.testing.assert_close(full_values.grad, chunk_values.grad)


class AnchorTest(unittest.TestCase):
    def test_chunks_are_reduced_then_one_global_logsumexp_is_used(self) -> None:
        first = torch.tensor(
            [[-0.2, -0.3, float("nan")], [-0.4, float("nan"), float("nan")]],
            requires_grad=True,
        )
        first_mask = torch.tensor([[True, True, False], [True, False, False]])
        second = torch.tensor(
            [[-0.7, -0.1, float("nan")]], requires_grad=True
        )
        second_mask = torch.tensor([[True, True, False]])

        seq_first, _ = gt_sequence_logps(first, first_mask)
        seq_second, _ = gt_sequence_logps(second, second_mask)
        all_sequences = torch.cat((seq_first, seq_second))
        loss = gt_set_anchor_loss(all_sequences)

        expected = -torch.logsumexp(torch.tensor([-0.5, -0.4, -0.8]), dim=0)
        torch.testing.assert_close(loss, expected)
        loss.backward()
        self.assertTrue(bool(torch.isfinite(first.grad[first_mask]).all()))
        self.assertTrue(bool(torch.isfinite(second.grad[second_mask]).all()))
        self.assertTrue(bool((first.grad[first_mask] < 0).all()))
        self.assertTrue(bool((second.grad[second_mask] < 0).all()))

    def test_single_fully_deterministic_gt_has_zero_anchor_loss(self) -> None:
        logps = torch.full((1, 2), float("nan"))
        mask = torch.zeros((1, 2), dtype=torch.bool)
        sequence, counts = gt_sequence_logps(logps, mask)
        self.assertEqual(counts.tolist(), [0])
        self.assertEqual(gt_set_anchor_loss(sequence).item(), 0.0)


class WindowAndCalibrationTest(unittest.TestCase):
    def test_branches_are_averaged_independently(self) -> None:
        output = window_loss(
            [torch.tensor(2.0), torch.tensor(4.0)],
            [torch.tensor(10.0) for _ in range(7)],
            anchor_weight=0.02,
            skip_groups=1,
        )
        self.assertAlmostEqual(output.rloo_mean.item(), 3.0)
        self.assertAlmostEqual(output.anchor_mean.item(), 10.0)
        self.assertAlmostEqual(output.weighted_anchor.item(), 0.2)
        self.assertAlmostEqual(output.loss.item(), 3.2)
        self.assertEqual((output.rloo_groups, output.anchor_groups, output.skip_groups), (2, 7, 1))

    def test_all_skip_window_has_no_optimizer_loss(self) -> None:
        output = window_loss([], [], anchor_weight=0.02, skip_groups=8)
        self.assertIsNone(output.loss)
        self.assertIsNone(output.rloo_mean)
        self.assertIsNone(output.anchor_mean)

    def test_calibration_formula_and_cap(self) -> None:
        self.assertAlmostEqual(calibrate_anchor_lambda(2.0, 10.0), 0.02)
        self.assertAlmostEqual(calibrate_anchor_lambda(10.0, 1.0), 0.05)
        with self.assertRaisesRegex(ValueError, "exceed"):
            calibrate_anchor_lambda(0.0, 1.0)
        with self.assertRaisesRegex(ValueError, "finite"):
            calibrate_anchor_lambda(math.nan, 1.0)


if __name__ == "__main__":
    unittest.main()
