from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from ksllm4rec_rloo.integrity import canonical_sha256
from ksllm4rec_rloo_dapo.config import approved_config
from ksllm4rec_rloo_dapo.gates import (
    expected_input_sha256,
    load_and_validate_gate_reports,
)


def _signature() -> dict:
    inputs = {"test": "gate"}
    return {"sha256": canonical_sha256(inputs), "inputs": inputs}


class GateValidationTest(unittest.TestCase):
    def _write_reports(self, root: Path, exact_speedup: float) -> dict[str, Path]:
        signature = _signature()
        reports = {
            "structure": {"input_sha256": expected_input_sha256()},
            "probability": {"max_canonical_replay_logp_difference": 0.0},
            "memory": {"peak_reserved_gib": 19.0},
            "throughput": {
                "cache_proposal_speedup": 3.0,
                "exact_cached_rollout_speedup": exact_speedup,
            },
            "pilot": {
                "parameter_changed": True,
                "end_to_end_window_speedup": 1.2,
            },
        }
        paths = {}
        for name, fields in reports.items():
            value = {
                "schema_version": 1,
                "gate": name,
                "passed": True,
                "runtime_signature_sha256": signature["sha256"],
                **fields,
            }
            path = root / f"{name}.json"
            path.write_text(json.dumps(value), encoding="utf-8")
            paths[name] = path
        return paths

    def test_exact_corrected_rollout_must_meet_the_speed_gate(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_reports(Path(directory), exact_speedup=1.49)
            with self.assertRaisesRegex(RuntimeError, "rollout threshold"):
                load_and_validate_gate_reports(
                    approved_config(), _signature(), paths
                )

    def test_complete_bound_reports_are_accepted(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_reports(Path(directory), exact_speedup=1.5)
            reports = load_and_validate_gate_reports(
                approved_config(), _signature(), paths
            )
            self.assertEqual(set(reports), set(paths))

    def test_end_to_end_window_speedup_is_mandatory(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self._write_reports(Path(directory), exact_speedup=1.5)
            pilot = json.loads(paths["pilot"].read_text(encoding="utf-8"))
            pilot["end_to_end_window_speedup"] = 1.19
            paths["pilot"].write_text(json.dumps(pilot), encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "end-to-end window"):
                load_and_validate_gate_reports(
                    approved_config(), _signature(), paths
                )


if __name__ == "__main__":
    unittest.main()
