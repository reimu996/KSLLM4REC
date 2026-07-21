from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ksllm4rec_rloo.config import approved_config
from ksllm4rec_rloo.gates import (
    GATE_SCHEMA_VERSION,
    GateBundle,
    expected_input_sha256,
    expected_structure,
    load_training_gates,
    projected_training_hours,
    validate_calibration_report,
    validate_memory_report,
    validate_probability_report,
    validate_signal_report,
    validate_structure_report,
    validate_timing_report,
)
from ksllm4rec_rloo.integrity import canonical_sha256


class GateValidationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = approved_config()
        inputs = {"test": "same-runtime"}
        self.signature = {"sha256": canonical_sha256(inputs), "inputs": inputs}
        self.lambda0 = 0.02
        self.resolved = {
            "spec_version": "2.0",
            "anchor": {"lambda0": self.lambda0},
            "reference": None,
        }

    def common(self, gate: str, *, resolved: bool = False) -> dict:
        value = {
            "schema_version": GATE_SCHEMA_VERSION,
            "gate": gate,
            "passed": True,
            "runtime_signature": self.signature,
        }
        if resolved:
            value.update(
                {
                    "resolved_contract": self.resolved,
                    "resolved_contract_sha256": canonical_sha256(self.resolved),
                    "lambda0": self.lambda0,
                }
            )
        return value

    def branches(self, groups: int, rloo: int | None = None) -> dict:
        rloo = groups // 2 if rloo is None else rloo
        return {
            "rloo_groups": rloo,
            "anchor_groups": groups - rloo,
            "skip_groups": 0,
            "rloo_group_rate": rloo / groups,
        }

    def legal(self, groups: int) -> dict:
        return {
            "candidates_run": groups * 16,
            "all_legal": True,
            "all_finite": True,
        }

    def chunks(self) -> dict:
        return {"rollout_chunk": 8, "loss_chunk": 8, "gt_chunk": 8}

    def structure_report(self) -> dict:
        structure = expected_structure(self.config)
        return {
            **self.common("structure"),
            "structure": structure,
            "structure_sha256": canonical_sha256(structure),
            "input_sha256": expected_input_sha256(),
        }

    def calibration_report(self) -> dict:
        groups = 512
        return {
            **self.common("calibration"),
            "groups_run": groups,
            **self.legal(groups),
            **self.branches(groups),
            "rloo_grad_norm": 2.0,
            "anchor_grad_norm": 10.0,
            "lambda0": self.lambda0,
            "resolved_contract": self.resolved,
            "resolved_contract_sha256": canonical_sha256(self.resolved),
            "optimizer_updates": 0,
            "parameter_max_change": 0.0,
        }

    def probability_report(self) -> dict:
        groups = 2
        return {
            **self.common("probability", resolved=True),
            "groups_run": groups,
            **self.legal(groups),
            "pre_update": True,
            "comparisons": 97,
            "max_abs_logp_difference": 1.0e-5,
        }

    def memory_report(self) -> dict:
        groups = 32
        return {
            **self.common("memory", resolved=True),
            "groups_run": groups,
            **self.legal(groups),
            **self.chunks(),
            "selection": "longest_prompt_tokens_desc_group_id_tiebreak",
            "group_ids": [f"g{index:02d}" for index in range(groups)],
            "prompt_token_lengths": list(range(1000, 968, -1)),
            "gpu_name": "NVIDIA GeForce RTX 4090",
            "peak_allocated_gib": 18.0,
            "peak_reserved_gib": 19.5,
            "backward_passes": 4,
            "optimizer_updates": 4,
        }

    def signal_report(self) -> dict:
        groups = 512
        return {
            **self.common("signal", resolved=True),
            "groups_run": groups,
            **self.legal(groups),
            **self.branches(groups),
            **self.chunks(),
            "optimizer_updates": 64,
            "gradient_norm_max": 1.5,
            "parameter_max_change": 0.0001,
            "loss_mean": 0.2,
        }

    def timing_report(self, seconds: float = 900.0) -> dict:
        groups = 256
        return {
            **self.common("timing", resolved=True),
            "groups_run": groups,
            **self.legal(groups),
            **self.branches(groups),
            **self.chunks(),
            "end_to_end": True,
            "seconds": seconds,
            "projected_two_epoch_hours": projected_training_hours(seconds, groups),
        }

    def test_structure_requires_exact_counts_and_locked_hashes(self) -> None:
        report = self.structure_report()
        validate_structure_report(self.config, report, self.signature)
        report["input_sha256"] = {**expected_input_sha256(), "groups": "0" * 64}
        with self.assertRaisesRegex(RuntimeError, "input SHA256"):
            validate_structure_report(self.config, report, self.signature)

    def test_calibration_recomputes_lambda_and_proves_no_update(self) -> None:
        resolved, value = validate_calibration_report(
            self.config, self.calibration_report(), self.signature
        )
        self.assertEqual(resolved, self.resolved)
        self.assertEqual(value, 0.02)
        report = self.calibration_report()
        report["optimizer_updates"] = 1
        with self.assertRaisesRegex(RuntimeError, "must not update"):
            validate_calibration_report(self.config, report, self.signature)
        report = self.calibration_report()
        report.update(self.branches(512, rloo=512))
        with self.assertRaisesRegex(RuntimeError, "anchor set is empty"):
            validate_calibration_report(self.config, report, self.signature)

    def test_probability_accepts_boundary_and_rejects_one_value_above(self) -> None:
        report = self.probability_report()
        validate_probability_report(
            self.config, report, self.signature, self.resolved
        )
        report["max_abs_logp_difference"] = math.nextafter(1.0e-5, math.inf)
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            validate_probability_report(
                self.config, report, self.signature, self.resolved
            )

    def test_memory_requires_longest_32_rtx4090_and_at_most_20_gib(self) -> None:
        report = self.memory_report()
        validate_memory_report(self.config, report, self.signature, self.resolved)
        report["peak_reserved_gib"] = 20.0001
        with self.assertRaisesRegex(RuntimeError, "above"):
            validate_memory_report(self.config, report, self.signature, self.resolved)

    def test_signal_requires_at_least_25_percent_rloo_groups(self) -> None:
        report = self.signal_report()
        validate_signal_report(self.config, report, self.signature, self.resolved)
        report.update(self.branches(512, rloo=127))
        with self.assertRaisesRegex(RuntimeError, "below"):
            validate_signal_report(self.config, report, self.signature, self.resolved)

    def test_timing_uses_256_groups_20_percent_margin_and_34032_visits(self) -> None:
        report = self.timing_report(seconds=900.0)
        self.assertAlmostEqual(
            report["projected_two_epoch_hours"],
            1.2 * (900.0 / 256 * 34032) / 3600,
        )
        validate_timing_report(self.config, report, self.signature, self.resolved)
        report["projected_two_epoch_hours"] /= 1.2
        with self.assertRaisesRegex(RuntimeError, "frozen formula"):
            validate_timing_report(self.config, report, self.signature, self.resolved)

    def test_every_candidate_must_be_legal_and_finite(self) -> None:
        report = self.signal_report()
        report["all_legal"] = False
        with self.assertRaisesRegex(RuntimeError, "all_legal"):
            validate_signal_report(self.config, report, self.signature, self.resolved)
        report = self.signal_report()
        report["loss_mean"] = float("nan")
        with self.assertRaisesRegex(RuntimeError, "finite"):
            validate_signal_report(self.config, report, self.signature, self.resolved)

    def test_load_training_gates_accepts_one_fully_bound_set(self) -> None:
        reports = {
            "structure": self.structure_report(),
            "calibration": self.calibration_report(),
            "probability": self.probability_report(),
            "memory": self.memory_report(),
            "signal": self.signal_report(),
            "timing": self.timing_report(),
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name, report in reports.items():
                path = root / f"{name}.json"
                path.write_text(json.dumps(report), encoding="utf-8")
                paths[name] = path
            with patch(
                "ksllm4rec_rloo.gates.runtime_signature",
                return_value=self.signature,
            ):
                bundle = load_training_gates(
                    config=self.config,
                    groups_path=Path("groups"),
                    trie_dir=Path("trie"),
                    calibration_ids_path=Path("calibration.json"),
                    structure_path=paths["structure"],
                    calibration_path=paths["calibration"],
                    probability_path=paths["probability"],
                    memory_path=paths["memory"],
                    signal_path=paths["signal"],
                    timing_path=paths["timing"],
                )
        self.assertIsInstance(bundle, GateBundle)
        self.assertEqual(bundle.lambda0, 0.02)
        self.assertEqual(
            (bundle.rollout_chunk, bundle.loss_chunk, bundle.gt_chunk), (8, 8, 8)
        )

    def test_load_rejects_report_from_another_runtime(self) -> None:
        report = self.signal_report()
        other_inputs = {"test": "other-runtime"}
        report["runtime_signature"] = {
            "sha256": canonical_sha256(other_inputs),
            "inputs": other_inputs,
        }
        with self.assertRaisesRegex(RuntimeError, "different runtime signature"):
            validate_signal_report(self.config, report, self.signature, self.resolved)

    def test_load_rejects_same_runtime_with_different_lambda(self) -> None:
        report = self.signal_report()
        different = {
            "spec_version": "2.0",
            "anchor": {"lambda0": 0.03},
            "reference": None,
        }
        report["resolved_contract"] = different
        report["resolved_contract_sha256"] = canonical_sha256(different)
        report["lambda0"] = 0.03
        with self.assertRaisesRegex(RuntimeError, "different resolved contract"):
            validate_signal_report(self.config, report, self.signature, self.resolved)


if __name__ == "__main__":
    unittest.main()
