from __future__ import annotations

import unittest

from ksllm4rec_dapo_anchor_v2_2.config import build_config
from ksllm4rec_dapo_anchor_v2_2.trainer import validate_calibration_report


class CalibrationReportTest(unittest.TestCase):
    def test_v21_calibration_report_is_rejected(self) -> None:
        config = build_config()
        report = {
            "schema_version": 2,
            "spec_version": "DAPO-ANCHOR-V2.1",
            "base_runtime_signature_sha256": "base",
            "requested_windows": 16,
            "valid_windows": 0,
            "windows": [],
            "lambda_calibrated": 0.0,
            "optimizer_updates": 0,
            "parameter_digest_before": "same",
            "parameter_digest_after": "same",
        }
        with self.assertRaisesRegex(ValueError, "V2.2"):
            validate_calibration_report(
                config,
                report,
                base_runtime_signature_sha256="base",
            )

    def test_lambda_is_recomputed_from_sixteen_zero_update_windows(self) -> None:
        config = build_config()
        windows = [
            {
                "window_index": index,
                "rl_minibatch_gradient_norms": [1.0, 2.0, 3.0, 4.0],
                "rl_reference_gradient_norm": 2.5,
                "anchor_raw_gradient_norm": 5.0,
                "anchor_candidate_groups": 48,
                "lambda_window": 0.05,
            }
            for index in range(16)
        ]
        report = {
            "schema_version": 2,
            "spec_version": "DAPO-ANCHOR-V2.2",
            "base_runtime_signature_sha256": "base",
            "requested_windows": 16,
            "valid_windows": 16,
            "windows": windows,
            "lambda_calibrated": 0.05,
            "optimizer_updates": 0,
            "parameter_digest_before": "same",
            "parameter_digest_after": "same",
        }
        self.assertEqual(
            validate_calibration_report(
                config,
                report,
                base_runtime_signature_sha256="base",
            ),
            0.05,
        )

    def test_parameter_change_invalidates_calibration(self) -> None:
        config = build_config()
        report = {
            "schema_version": 2,
            "spec_version": "DAPO-ANCHOR-V2.2",
            "base_runtime_signature_sha256": "base",
            "requested_windows": 16,
            "valid_windows": 0,
            "windows": [
                {
                    "window_index": index,
                    "rl_minibatch_gradient_norms": [0.0, 0.0, 0.0, 0.0],
                    "rl_reference_gradient_norm": 0.0,
                    "anchor_raw_gradient_norm": 0.0,
                    "anchor_candidate_groups": 0,
                    "lambda_window": None,
                }
                for index in range(16)
            ],
            "lambda_calibrated": 0.0,
            "optimizer_updates": 0,
            "parameter_digest_before": "before",
            "parameter_digest_after": "after",
        }
        with self.assertRaisesRegex(RuntimeError, "changed"):
            validate_calibration_report(
                config,
                report,
                base_runtime_signature_sha256="base",
            )


if __name__ == "__main__":
    unittest.main()
