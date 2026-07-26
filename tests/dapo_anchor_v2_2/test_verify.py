from __future__ import annotations

import unittest

from ksllm4rec_dapo_anchor_v2_2 import verify


def _window(
    *,
    anchor_groups: int,
    anchor_steps: int,
    ratio: float,
    window_index: int = 0,
) -> dict:
    return {
        "window_index": window_index,
        "effective_groups": 32,
        "filter_groups": anchor_groups,
        "anchor_candidate_groups": anchor_groups,
        "anchor_groups_used": anchor_groups,
        "anchor_group_ids": [f"g{index}" for index in range(anchor_groups)],
        "anchor_gt_sid_count": anchor_groups,
        "anchor_decision_token_count": anchor_groups * 3,
        "anchor_set_nll_mean": 1.0,
        "anchor_all_gt_decision_nll": 1.0,
        "anchor_phase_seconds": 1.0,
        "anchor_raw_grad_norm": 4.0 if anchor_steps else 0.0,
        "anchor_post_scale_grad_norm": 0.005 if anchor_steps else 0.0,
        "rl_reference_grad_norm": 0.05,
        "lambda_calibrated": 0.01,
        "lambda_cap": 0.00125 if anchor_steps else 0.0,
        "lambda_effective": 0.00125 if anchor_steps else 0.0,
        "anchor_to_rl_grad_ratio": ratio,
        "anchor_max_replay_logp_difference": 0.0,
        "rl_optimizer_steps": 4,
        "anchor_optimizer_steps": anchor_steps,
        "optimizer_updates": 4 + anchor_steps,
        "optimizer_update_step": (window_index + 1) * 4 + anchor_steps,
        "steps": [
            {"group_ids": [f"r{step_index}_{index}" for index in range(8)]}
            for step_index in range(4)
        ],
    }


class V22WindowVerifierTest(unittest.TestCase):
    def test_w0_anchor_step_is_valid(self) -> None:
        verify.validate_window_log_row(
            _window(anchor_groups=2, anchor_steps=1, ratio=0.10)
        )

    def test_w9_uses_all_48_anchor_groups(self) -> None:
        verify.validate_window_log_row(
            _window(
                anchor_groups=48,
                anchor_steps=1,
                ratio=0.10,
                window_index=9,
            )
        )

    def test_all_129_anchor_groups_are_valid_and_single_step(self) -> None:
        verify.validate_window_log_row(_window(anchor_groups=129, anchor_steps=1, ratio=0.10))

    def test_more_than_one_anchor_step_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "anchor optimizer"):
            verify.validate_window_log_row(_window(anchor_groups=48, anchor_steps=2, ratio=0.10))

    def test_anchor_gradient_ratio_over_ten_percent_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "anchor_to_rl"):
            verify.validate_window_log_row(_window(anchor_groups=48, anchor_steps=1, ratio=0.100001))

    def test_all_candidates_may_be_measured_when_zero_lambda_skips_step(self) -> None:
        verify.validate_window_log_row(
            _window(anchor_groups=129, anchor_steps=0, ratio=0.0)
        )

    def test_in_memory_tuple_group_ids_are_accepted_before_json_serialization(self) -> None:
        row = _window(anchor_groups=2, anchor_steps=1, ratio=0.10)
        for step in row["steps"]:
            step["group_ids"] = tuple(step["group_ids"])
        verify.validate_window_log_row(row)


if __name__ == "__main__":
    unittest.main()
