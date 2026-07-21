from __future__ import annotations

import hashlib
import unittest

from ksllm4rec_grpo.data import RecommendationGroup
from ksllm4rec_rloo.config import approved_config
from ksllm4rec_rloo.trainer import (
    _summarize_audit,
    calibration_rank,
    expected_calibration_ids,
    learning_rate_for_window,
)


class ScheduleTest(unittest.TestCase):
    def test_window_schedule_matches_the_frozen_formula(self) -> None:
        config = approved_config()
        maximum = config["train"]["learning_rate"]
        self.assertAlmostEqual(learning_rate_for_window(config, 0), maximum / 128)
        self.assertAlmostEqual(learning_rate_for_window(config, 127), maximum)
        self.assertAlmostEqual(learning_rate_for_window(config, 128), maximum)
        self.assertAlmostEqual(learning_rate_for_window(config, 4253), 0.0)

    def test_invalid_window_is_rejected(self) -> None:
        config = approved_config()
        for value in (-1, 4254):
            with self.assertRaisesRegex(ValueError, "window_index"):
                learning_rate_for_window(config, value)


class CalibrationSelectionTest(unittest.TestCase):
    @staticmethod
    def _group(group_id: str) -> RecommendationGroup:
        return RecommendationGroup(group_id, "s", "p/no_think", ("sid",), (1,))

    def test_rank_uses_the_exact_payload(self) -> None:
        group_id = "abc"
        expected = hashlib.sha256(
            b"anchor-calibration|42|abc"
        ).hexdigest()
        self.assertEqual(calibration_rank(group_id), expected)

    def test_selection_is_hash_ranked_not_source_ordered(self) -> None:
        groups = [self._group(f"group-{index}") for index in range(8)]
        actual = expected_calibration_ids(groups, count=3)
        expected = [
            group_id
            for _, group_id in sorted(
                (calibration_rank(group.group_id), group.group_id) for group in groups
            )[:3]
        ]
        self.assertEqual(actual, expected)


class AuditSummaryTest(unittest.TestCase):
    def test_reports_live_reward_without_any_gt_injection(self) -> None:
        rows = [
            {
                "epoch": 1,
                "branch": "rloo",
                "reward_tiers": ["exact"] + ["other_domain"] * 15,
                "reward_mean_live": 1.0 / 16,
                "gt_injection_count": 0,
            },
            {
                "epoch": 1,
                "branch": "gt_set_anchor",
                "reward_tiers": ["same_domain"] * 16,
                "reward_mean_live": 0.01,
                "gt_injection_count": 0,
            },
        ]
        summary = _summarize_audit(rows)
        self.assertEqual(summary["groups"], 2)
        self.assertEqual(summary["gt_injection_count"], 0)
        self.assertEqual(summary["branch_counts"], {"rloo": 1, "gt_set_anchor": 1})
        self.assertAlmostEqual(summary["candidate_exact_rate"], 1 / 32)


if __name__ == "__main__":
    unittest.main()
