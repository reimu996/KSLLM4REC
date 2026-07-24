from __future__ import annotations

import math
import unittest

import torch

from ksllm4rec_rloo_dapo.objective import (
    clipped_rloo_token_sum,
    is_effective_rewards,
    rloo_advantages,
)


class EffectiveRewardTest(unittest.TestCase):
    def test_every_equal_reward_tier_is_rejected(self) -> None:
        for value in (0.0, 0.01, 0.15, 0.40, 1.0):
            with self.subTest(value=value):
                self.assertFalse(is_effective_rewards(torch.full((16,), value)))

    def test_any_reward_difference_is_effective(self) -> None:
        rewards = torch.full((16,), 0.01)
        rewards[-1] = 0.15
        self.assertTrue(is_effective_rewards(rewards))
        advantages = rloo_advantages(rewards)
        self.assertAlmostEqual(float(advantages.sum()), 0.0, places=6)


class AsymmetricClipTest(unittest.TestCase):
    def test_positive_and_negative_advantages_clip_on_opposite_sides(self) -> None:
        ratios = torch.tensor([1.5, 0.5, 0.5, 1.5])
        output = clipped_rloo_token_sum(
            ratios.log().reshape(4, 1),
            torch.zeros((4, 1)),
            torch.tensor([1.0, -1.0, 1.0, -1.0]),
            torch.ones((4, 1), dtype=torch.bool),
        )
        self.assertAlmostEqual(float(output.loss_sum), 0.52, places=5)
        self.assertEqual(output.clip_active.tolist(), [True, True, False, False])
        self.assertEqual(output.below_low.tolist(), [False, True, True, False])
        self.assertEqual(output.above_high.tolist(), [True, False, False, True])

    def test_ratio_one_matches_unclipped_policy_gradient(self) -> None:
        advantages = torch.tensor([0.25, -0.25])
        output = clipped_rloo_token_sum(
            torch.zeros((2, 2)),
            torch.zeros((2, 2)),
            advantages,
            torch.ones((2, 2), dtype=torch.bool),
        )
        self.assertTrue(math.isclose(float(output.loss_sum), 0.0, abs_tol=1e-7))
        self.assertFalse(bool(output.clip_active.any()))


if __name__ == "__main__":
    unittest.main()
