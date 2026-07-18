from __future__ import annotations

import hashlib
import math
import unittest

import torch

from ksllm4rec_grpo.objective import (
    deterministic_uniform,
    finalize_candidates,
    grpo_loss,
    rewards_and_advantages,
    select_forced_gt,
)
from ksllm4rec_orpo.data import Sid


GT = Sid("video", 1, 2, 3)


def _live_candidates() -> list[Sid]:
    return [Sid("ad", 100 + index, 200 + index, 300 + index) for index in range(8)]


class ForcedGtTest(unittest.TestCase):
    def test_hash_matches_frozen_payload_and_is_repeatable(self) -> None:
        payload = b"add_gt|42|1|group-alpha"
        expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") / 2**64
        first = deterministic_uniform("add_gt", group_id="group-alpha", epoch_index=1)
        second = deterministic_uniform("add_gt", group_id="group-alpha", epoch_index=1)
        self.assertEqual(first, expected)
        self.assertEqual(first, second)

    def test_existing_live_gt_prevents_replacement_without_hash_decision(self) -> None:
        live = _live_candidates()
        live[2] = GT
        result = finalize_candidates(
            live,
            [GT],
            group_id="group-that-would-otherwise-add",
            epoch_index=1,
            probability=1.0,
        )
        self.assertTrue(result.has_live_gt)
        self.assertFalse(result.add_gt)
        self.assertIsNone(result.add_gt_uniform)
        self.assertEqual(result.final_candidates, tuple(live))
        self.assertEqual(result.forced_mask, (False,) * 8)
        self.assertIsNone(result.discarded_rollout)

    def test_no_gt_and_true_hash_replaces_index_seven_and_audits_discard(self) -> None:
        live = _live_candidates()
        positives = [
            Sid("video", 9, 8, 7),
            GT,
            Sid("prod", 4, 5, 6),
        ]
        group_id = "add-group-1"
        result = finalize_candidates(live, positives, group_id=group_id, epoch_index=2)
        expected_gt = select_forced_gt(positives, group_id=group_id, epoch_index=2)
        ordered = sorted(positives, key=Sid.render)
        selection_payload = f"gt_index|42|2|{group_id}".encode()
        selection_number = int.from_bytes(
            hashlib.sha256(selection_payload).digest()[:8], "big"
        )
        independently_selected_gt = ordered[selection_number % len(ordered)]
        self.assertFalse(result.has_live_gt)
        self.assertTrue(result.add_gt)
        self.assertEqual(result.final_candidates[:7], tuple(live[:7]))
        self.assertEqual(result.final_candidates[7], expected_gt)
        self.assertEqual(expected_gt, independently_selected_gt)
        self.assertEqual(result.forced_mask, (False,) * 7 + (True,))
        self.assertEqual(result.forced_gt, expected_gt)
        self.assertEqual(result.discarded_rollout, live[7])

    def test_no_gt_and_false_hash_keeps_all_live_candidates(self) -> None:
        live = _live_candidates()
        group_id = "keep-group-0"
        result = finalize_candidates(live, [GT], group_id=group_id, epoch_index=1)
        self.assertFalse(result.has_live_gt)
        self.assertFalse(result.add_gt)
        self.assertEqual(result.final_candidates, tuple(live))
        self.assertIsNotNone(result.add_gt_uniform)
        self.assertIsNone(result.discarded_rollout)

    def test_rejects_any_group_that_did_not_first_generate_eight(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly 8"):
            finalize_candidates(
                _live_candidates()[:7],
                [GT],
                group_id="short-group",
                epoch_index=1,
            )


class RewardAndAdvantageTest(unittest.TestCase):
    def test_all_reward_tiers_and_sample_zscore_match_hand_calculation(self) -> None:
        candidates = [
            GT,
            Sid("video", 1, 2, 9),
            Sid("video", 1, 8, 9),
            Sid("video", 8, 8, 9),
            Sid("ad", 1, 2, 3),
            Sid("video", 1, 2, 10),
            Sid("video", 7, 7, 7),
            Sid("prod", 7, 7, 7),
        ]
        result = rewards_and_advantages(candidates, [GT])
        expected_rewards = torch.tensor(
            [1.0, 0.20, 0.05, 0.01, 0.0, 0.20, 0.01, 0.0],
            dtype=torch.float32,
        )
        expected_mean = expected_rewards.sum() / 8
        squared_deviations = ((expected_rewards - expected_mean) ** 2).sum()
        expected_std = torch.sqrt(squared_deviations / 7)
        expected_advantages = (expected_rewards - expected_mean) / (
            expected_std + 1.0e-4
        )
        torch.testing.assert_close(result.rewards, expected_rewards)
        torch.testing.assert_close(result.mean_reward, expected_mean)
        torch.testing.assert_close(result.std_reward, expected_std)
        torch.testing.assert_close(result.advantages, expected_advantages)
        self.assertEqual(
            result.tiers,
            (
                "exact",
                "same_ab",
                "same_a",
                "same_domain",
                "other_domain",
                "same_ab",
                "same_domain",
                "other_domain",
            ),
        )

    def test_forced_gt_example_uses_correction_one(self) -> None:
        rewards = torch.tensor([0.20, 0.20, 0.05, 0.01, 0, 0, 0, 1.0])
        candidates = [
            Sid("video", 1, 2, 10),
            Sid("video", 1, 2, 11),
            Sid("video", 1, 8, 10),
            Sid("video", 8, 8, 10),
            Sid("ad", 8, 8, 10),
            Sid("prod", 8, 8, 10),
            Sid("living", 8, 8, 10),
            GT,
        ]
        result = rewards_and_advantages(candidates, [GT])
        self.assertAlmostEqual(result.mean_reward.item(), 0.1825, places=6)
        self.assertAlmostEqual(
            result.std_reward.item(), math.sqrt(0.81615 / 7), places=6
        )
        torch.testing.assert_close(result.rewards, rewards)

    def test_equal_rewards_produce_zero_finite_advantages(self) -> None:
        candidates = [Sid("ad", index, index, index) for index in range(8)]
        result = rewards_and_advantages(candidates, [GT])
        self.assertEqual(result.std_reward.item(), 0.0)
        torch.testing.assert_close(result.advantages, torch.zeros(8))
        self.assertTrue(bool(torch.isfinite(result.advantages).all()))

    def test_equal_nonzero_float_rewards_are_exactly_zeroed(self) -> None:
        candidates = [Sid("video", 100 + index, index, index) for index in range(8)]
        result = rewards_and_advantages(candidates, [GT])
        torch.testing.assert_close(result.rewards, torch.full((8,), 0.01))
        self.assertEqual(result.std_reward.item(), 0.0)
        torch.testing.assert_close(result.advantages, torch.zeros(8))


class GrpoLossTest(unittest.TestCase):
    def test_clipped_live_forced_surrogate_and_kl_match_literal_formula(self) -> None:
        new = torch.tensor(
            [
                [-0.10, -0.30, float("nan")],
                [-0.40, float("nan"), float("nan")],
                [-0.20, -0.25, -0.35],
            ],
            dtype=torch.float32,
            requires_grad=True,
        )
        reference = torch.tensor(
            [
                [-0.15, -0.25, float("nan")],
                [-0.45, float("nan"), float("nan")],
                [-0.20, -0.30, float("nan")],
            ]
        )
        decision_mask = torch.tensor(
            [[True, True, False], [True, False, False], [True, True, False]]
        )
        forced_mask = torch.tensor([False, False, True])
        old = torch.tensor(
            [
                [-0.40, -0.30, float("nan")],
                [-0.10, float("nan"), float("nan")],
            ]
        )
        advantages = torch.tensor([1.0, -1.0, 2.0])

        output = grpo_loss(
            new,
            reference,
            old,
            advantages,
            decision_mask,
            forced_mask,
        )

        ratio_00 = math.exp(0.30)
        term_00 = min(ratio_00, 1.20)
        term_01 = 1.0
        ratio_10 = math.exp(-0.30)
        term_10 = min(-ratio_10, -0.80)
        forced_term = 2.0

        def kl(ref: float, current: float) -> float:
            d = ref - current
            return math.exp(d) - d - 1.0

        expected_per_candidate = torch.tensor(
            [
                -((term_00 + term_01) / 2)
                + 0.02 * ((kl(-0.15, -0.10) + kl(-0.25, -0.30)) / 2),
                -term_10 + 0.02 * kl(-0.45, -0.40),
                -forced_term + 0.02 * ((kl(-0.20, -0.20) + kl(-0.30, -0.25)) / 2),
            ]
        )
        torch.testing.assert_close(
            output.per_candidate_loss, expected_per_candidate, atol=2e-6, rtol=0
        )
        torch.testing.assert_close(output.loss, expected_per_candidate.mean())
        self.assertTrue(bool(torch.isfinite(output.loss)))

        output.loss.backward()
        self.assertTrue(bool(torch.isfinite(new.grad[decision_mask]).all()))
        self.assertLess(new.grad[2, 0].item(), 0.0)
        self.assertLess(new.grad[2, 1].item(), 0.0)

    def test_forced_gt_has_no_old_policy_row(self) -> None:
        new = torch.full((8, 2), -0.5, requires_grad=True)
        reference = torch.full((8, 2), -0.5)
        mask = torch.ones((8, 2), dtype=torch.bool)
        forced = torch.tensor([False] * 7 + [True])
        advantages = torch.arange(8, dtype=torch.float32)
        valid_old = torch.full((7, 2), -0.5)
        result = grpo_loss(new, reference, valid_old, advantages, mask, forced)
        self.assertTrue(bool(torch.isfinite(result.loss)))
        with self.assertRaisesRegex(ValueError, "old_logps"):
            grpo_loss(
                new,
                reference,
                torch.full((8, 2), -0.5),
                advantages,
                mask,
                forced,
            )


if __name__ == "__main__":
    unittest.main()
