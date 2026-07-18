from __future__ import annotations

import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from ksllm4rec_sft.gates import REQUIRED_GATES, verify_gpu_gates
from ksllm4rec_sft.profiles import FRONTIER_PROFILE, MODEL_PATH


def frontier_integrity() -> dict:
    return {
        "status": "passed",
        "profile": FRONTIER_PROFILE.name,
        "lock_path": str(FRONTIER_PROFILE.artifact_lock_path),
        "lock_sha256": "current-lock",
        "model_root": str(MODEL_PATH),
        "model_files": {},
        "source_dataset": {
            "path": str(FRONTIER_PROFILE.source_path),
            "size": FRONTIER_PROFILE.source_size,
            "sha256": FRONTIER_PROFILE.source_sha256,
            "records": FRONTIER_PROFILE.source_records,
        },
        "derived_dataset": {
            "path": str(FRONTIER_PROFILE.derived_path),
            "size": 123,
            "sha256": "derived-sha",
            "records": FRONTIER_PROFILE.source_records,
        },
        "dataset_info": {
            "path": str(FRONTIER_PROFILE.dataset_dir / "dataset_info.json"),
            "size": 10,
            "sha256": "dataset-info-sha",
        },
        "llamafactory": {"path": "/lf", "commit": "c", "tree": "t"},
    }


class GpuGateValidationTest(unittest.TestCase):
    def test_requires_exactly_the_four_approved_lengths(self) -> None:
        self.assertEqual(
            REQUIRED_GATES,
            {
                "gate_00512": 512,
                "gate_02048": 2048,
                "gate_08192": 8192,
                "gate_16384": 16384,
            },
        )

    def test_accepts_one_valid_manifest_per_required_length(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, (stage, cutoff) in enumerate(REQUIRED_GATES.items()):
                run = root / f"run-{index}"
                run.mkdir()
                manifest = {
                    "status": "passed",
                    "stage": stage,
                    "updated_at": f"2026-07-14T00:00:0{index}+08:00",
                    "resolved_config": {"cutoff_len": cutoff, "max_steps": 1},
                    "custom_loss": {
                        "gamma": 2.0,
                        "item_weight": 3.0,
                        "chunk_size": 512,
                    },
                    "result": {
                        "optimizer_steps": 1,
                        "fallback_count": 0,
                        "max_sequence_length": cutoff,
                    },
                    "implementation_fingerprint": {"sha256": "current"},
                    "peak_memory_reserved_gib": 20.0,
                }
                (run / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
            with patch(
                "ksllm4rec_sft.gates.implementation_fingerprint",
                return_value={"sha256": "current", "files": {}},
            ):
                report = verify_gpu_gates(root, root / "report.json", root)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(set(report["accepted"]), set(REQUIRED_GATES))

    def test_rejects_focal_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, (stage, cutoff) in enumerate(REQUIRED_GATES.items()):
                run = root / f"run-{index}"
                run.mkdir()
                manifest = {
                    "status": "passed",
                    "stage": stage,
                    "resolved_config": {"cutoff_len": cutoff, "max_steps": 1},
                    "custom_loss": {
                        "gamma": 2.0,
                        "item_weight": 3.0,
                        "chunk_size": 512,
                    },
                    "result": {
                        "optimizer_steps": 1,
                        "fallback_count": int(index == 0),
                        "max_sequence_length": cutoff,
                    },
                    "implementation_fingerprint": {"sha256": "current"},
                    "peak_memory_reserved_gib": 20.0,
                }
                (run / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
            with (
                patch(
                    "ksllm4rec_sft.gates.implementation_fingerprint",
                    return_value={"sha256": "current", "files": {}},
                ),
                self.assertRaisesRegex(RuntimeError, "gate_00512"),
            ):
                verify_gpu_gates(root, root / "report.json", root)

    def test_rejects_stale_implementation_fingerprint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, (stage, cutoff) in enumerate(REQUIRED_GATES.items()):
                run = root / f"run-{index}"
                run.mkdir()
                manifest = {
                    "status": "passed",
                    "stage": stage,
                    "resolved_config": {"cutoff_len": cutoff, "max_steps": 1},
                    "custom_loss": {
                        "gamma": 2.0,
                        "item_weight": 3.0,
                        "chunk_size": 512,
                    },
                    "result": {
                        "optimizer_steps": 1,
                        "fallback_count": 0,
                        "max_sequence_length": cutoff,
                    },
                    "implementation_fingerprint": {
                        "sha256": "stale" if index == 0 else "current"
                    },
                    "peak_memory_reserved_gib": 20.0,
                }
                (run / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
            with (
                patch(
                    "ksllm4rec_sft.gates.implementation_fingerprint",
                    return_value={"sha256": "current", "files": {}},
                ),
                self.assertRaisesRegex(RuntimeError, "gate_00512"),
            ):
                verify_gpu_gates(root, root / "report.json", root)

    def test_rejects_stage_that_did_not_reach_requested_length(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, (stage, cutoff) in enumerate(REQUIRED_GATES.items()):
                run = root / f"run-{index}"
                run.mkdir()
                observed_length = cutoff // 2 if index == 0 else cutoff
                manifest = {
                    "status": "passed",
                    "stage": stage,
                    "resolved_config": {"cutoff_len": cutoff, "max_steps": 1},
                    "custom_loss": {
                        "gamma": 2.0,
                        "item_weight": 3.0,
                        "chunk_size": 512,
                    },
                    "result": {
                        "optimizer_steps": 1,
                        "fallback_count": 0,
                        "max_sequence_length": observed_length,
                    },
                    "implementation_fingerprint": {"sha256": "current"},
                    "peak_memory_reserved_gib": 20.0,
                }
                (run / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
            with (
                patch(
                    "ksllm4rec_sft.gates.implementation_fingerprint",
                    return_value={"sha256": "current", "files": {}},
                ),
                self.assertRaisesRegex(RuntimeError, "gate_00512"),
            ):
                verify_gpu_gates(root, root / "report.json", root)

    def test_rejects_gate_run_with_a_smaller_chunk_override(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for index, (stage, cutoff) in enumerate(REQUIRED_GATES.items()):
                run = root / f"run-{index}"
                run.mkdir()
                manifest = {
                    "status": "passed",
                    "stage": stage,
                    "resolved_config": {"cutoff_len": cutoff, "max_steps": 1},
                    "custom_loss": {
                        "gamma": 2.0,
                        "item_weight": 3.0,
                        "chunk_size": 128 if index == 0 else 512,
                    },
                    "result": {
                        "optimizer_steps": 1,
                        "fallback_count": 0,
                        "max_sequence_length": cutoff,
                    },
                    "implementation_fingerprint": {"sha256": "current"},
                    "peak_memory_reserved_gib": 20.0,
                }
                (run / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
            with (
                patch(
                    "ksllm4rec_sft.gates.implementation_fingerprint",
                    return_value={"sha256": "current", "files": {}},
                ),
                self.assertRaisesRegex(RuntimeError, "gate_00512"),
            ):
                verify_gpu_gates(root, root / "report.json", root)

    def test_frontier_accepts_only_its_profile_and_exact_artifact_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            integrity = frontier_integrity()
            for index, (stage, cutoff) in enumerate(
                FRONTIER_PROFILE.gate_cutoffs.items()
            ):
                run = root / f"frontier-{index}"
                run.mkdir()
                manifest = {
                    "status": "passed",
                    "profile": FRONTIER_PROFILE.name,
                    "stage": stage,
                    "updated_at": f"2026-07-19T00:00:0{index}+08:00",
                    "resolved_config": {
                        "dataset": FRONTIER_PROFILE.dataset_name,
                        "dataset_dir": str(FRONTIER_PROFILE.dataset_dir),
                        "cutoff_len": cutoff,
                        "max_steps": 1,
                    },
                    "custom_loss": {
                        "gamma": 2.0,
                        "item_weight": 3.0,
                        "chunk_size": 512,
                    },
                    "result": {
                        "optimizer_steps": 1,
                        "fallback_count": 0,
                        "max_sequence_length": cutoff,
                    },
                    "input_integrity": deepcopy(integrity),
                    "implementation_fingerprint": {"sha256": "current"},
                    "peak_memory_reserved_gib": 20.0,
                }
                (run / "manifest.json").write_text(
                    json.dumps(manifest), encoding="utf-8"
                )
                wrong = root / f"wrong-lock-{index}"
                wrong.mkdir()
                wrong_manifest = deepcopy(manifest)
                wrong_manifest["updated_at"] = f"2026-07-20T00:00:0{index}+08:00"
                wrong_manifest["input_integrity"]["lock_sha256"] = "wrong-lock"
                (wrong / "manifest.json").write_text(
                    json.dumps(wrong_manifest), encoding="utf-8"
                )
            with patch(
                "ksllm4rec_sft.gates.implementation_fingerprint",
                return_value={"sha256": "current", "files": {}},
            ):
                report = verify_gpu_gates(
                    root,
                    root / "report.json",
                    root,
                    profile=FRONTIER_PROFILE,
                    expected_input_integrity=integrity,
                )
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["profile"], FRONTIER_PROFILE.name)
            self.assertEqual(
                set(report["accepted"]), set(FRONTIER_PROFILE.gate_cutoffs)
            )
            for accepted in report["accepted"].values():
                self.assertIn("frontier-", accepted["manifest"])

    def test_frontier_gate_check_requires_the_current_artifact_lock(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(RuntimeError, "artifact lock identity"):
                verify_gpu_gates(
                    root, root / "report.json", root, profile=FRONTIER_PROFILE
                )


if __name__ == "__main__":
    unittest.main()
