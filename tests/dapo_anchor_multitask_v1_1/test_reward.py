from __future__ import annotations

import unittest

import torch

from ksllm4rec_dapo_anchor_multitask_v1_1.reward import (
    ObjectiveRoute,
    REWARD_VALUES,
    RewardTier,
    route_group,
    score_group,
)
from ksllm4rec_dapo_anchor_multitask_v1_1.objective import (
    group_relative_rewards_and_advantages,
)
from ksllm4rec_dapo_anchor_multitask_v1_1._infra.orpo_data import Sid


GT = Sid("video", 1, 2, 3)


class SharedRewardTest(unittest.TestCase):
    def test_five_frozen_tiers_and_highest_multi_gt_match(self) -> None:
        second = Sid("prod", 7, 8, 9)
        candidates = [
            GT,
            Sid("video", 1, 2, 99),
            Sid("video", 1, 99, 99),
            Sid("video", 99, 99, 99),
            Sid("ad", 1, 2, 3),
            second,
        ] + [Sid("living", 100 + index, index, index) for index in range(10)]
        result = score_group(candidates, [GT, second])
        self.assertEqual(
            result.rewards[:6],
            (1.0, 0.40, 0.15, 0.01, 0.0, 1.0),
        )
        self.assertEqual(
            result.tiers[:6],
            (
                RewardTier.EXACT,
                RewardTier.SAME_AB,
                RewardTier.SAME_A,
                RewardTier.SAME_DOMAIN,
                RewardTier.OTHER_DOMAIN,
                RewardTier.EXACT,
            ),
        )
        self.assertEqual(
            dict(REWARD_VALUES),
            {
                "exact": 1.0,
                "same_ab": 0.40,
                "same_a": 0.15,
                "same_domain": 0.01,
                "other_domain": 0.0,
            },
        )

    def test_no_exact_varied_routes_to_both_rl_and_anchor(self) -> None:
        candidates = [Sid("video", 1, 2, 100)] + [
            Sid("ad", 100 + index, index, index) for index in range(15)
        ]
        decision = route_group(score_group(candidates, [GT]))
        self.assertEqual(decision.route, ObjectiveRoute.RL_AND_ANCHOR)
        self.assertTrue(decision.use_rl)
        self.assertTrue(decision.use_anchor)

    def test_no_exact_flat_routes_only_to_anchor(self) -> None:
        candidates = [Sid("ad", 100 + index, index, index) for index in range(16)]
        decision = route_group(score_group(candidates, [GT]))
        self.assertEqual(decision.route, ObjectiveRoute.ANCHOR)
        self.assertFalse(decision.use_rl)
        self.assertTrue(decision.use_anchor)

    def test_exact_varied_routes_to_rl_and_anchor(self) -> None:
        candidates = [GT] + [
            Sid("ad", 100 + index, index, index) for index in range(15)
        ]
        decision = route_group(score_group(candidates, [GT]))
        self.assertEqual(decision.route, ObjectiveRoute.RL_AND_ANCHOR)
        self.assertTrue(decision.use_rl)
        self.assertTrue(decision.use_anchor)

    def test_all_exact_routes_to_anchor_instead_of_being_skipped(self) -> None:
        decision = route_group(score_group([GT] * 16, [GT]))
        self.assertEqual(decision.route, ObjectiveRoute.ANCHOR)
        self.assertFalse(decision.use_rl)
        self.assertTrue(decision.use_anchor)

    def test_training_objective_uses_the_same_reward_table(self) -> None:
        candidates = [
            Sid("video", 1, 2, 100),
            Sid("video", 1, 100, 100),
            Sid("video", 100, 100, 100),
            Sid("ad", 100, 100, 100),
        ] * 4
        output = group_relative_rewards_and_advantages(candidates, [GT])
        torch.testing.assert_close(
            output.rewards[:4], torch.tensor([0.40, 0.15, 0.01, 0.0])
        )
        self.assertAlmostEqual(float(output.advantages.sum()), 0.0, places=7)

    def test_training_objective_uses_the_selected_experiment_reward_profile(self) -> None:
        candidates = [Sid("video", 1, 2, 100)] * 16
        old = group_relative_rewards_and_advantages(
            candidates,
            [GT],
            reward_values={
                "exact": 1.0,
                "same_ab": 0.40,
                "same_a": 0.15,
                "same_domain": 0.01,
                "other_domain": 0.0,
            },
        )
        new = group_relative_rewards_and_advantages(
            candidates,
            [GT],
            reward_values={
                "exact": 1.0,
                "same_ab": 0.15,
                "same_a": 0.05,
                "same_domain": 0.01,
                "other_domain": 0.0,
            },
        )
        self.assertAlmostEqual(float(old.rewards[0]), 0.40, places=6)
        self.assertAlmostEqual(float(new.rewards[0]), 0.15, places=6)


if __name__ == "__main__":
    unittest.main()
