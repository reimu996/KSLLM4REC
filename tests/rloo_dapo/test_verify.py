from __future__ import annotations

from copy import deepcopy
import unittest

from ksllm4rec_rloo_dapo.config import approved_config
from ksllm4rec_rloo_dapo.trainer import learning_rate_for_window
from ksllm4rec_rloo_dapo.verify import validate_window_rows


def _window(config, index: int):
    ids = [f"w{index}-g{group}" for group in range(32)]
    lr = learning_rate_for_window(config, index)
    steps = []
    for start in range(0, 32, 8):
        steps.append(
            {
                "group_ids": ids[start : start + 8],
                "learning_rate": lr,
                "max_replay_logp_difference": 0.0,
                "ratio_min": 0.9,
                "ratio_max": 1.1,
                "ratio_mean": 1.0,
            }
        )
    return {
        "window_index": index,
        "learning_rate": lr,
        "group_ids": ids,
        "steps": steps,
        "peak_reserved_gib": 10.0,
        "anchor_groups": 0,
        "gt_injection_count": 0,
        "k": 1,
    }


class VerifyTest(unittest.TestCase):
    def test_two_complete_windows_have_eight_updates(self) -> None:
        config = approved_config()
        result = validate_window_rows(
            config, [_window(config, 0), _window(config, 1)], expected_windows=2
        )
        self.assertEqual(result["optimizer_updates"], 8)
        self.assertEqual(result["group_occurrences"], 64)

    def test_duplicate_group_in_one_window_is_rejected(self) -> None:
        config = approved_config()
        row = _window(config, 0)
        row["group_ids"][-1] = row["group_ids"][0]
        with self.assertRaisesRegex(RuntimeError, "K=1"):
            validate_window_rows(config, [row], expected_windows=1)


if __name__ == "__main__":
    unittest.main()
