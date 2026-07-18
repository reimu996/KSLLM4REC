from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from ksllm4rec_sft.checkpoint import (
    CHECKPOINT_BINDING_FILE,
    REQUIRED_CHECKPOINT_FILES,
    resolve_resume_checkpoint,
    write_checkpoint_binding,
    write_run_binding,
    CheckpointBindingCallback,
)
from ksllm4rec_sft.integrity import artifact_identity


def write_checkpoint(root: Path, step: int, *, omit: str | None = None) -> Path:
    checkpoint = root / f"checkpoint-{step}"
    checkpoint.mkdir()
    for name in REQUIRED_CHECKPOINT_FILES:
        if name == omit:
            continue
        if name == "trainer_state.json":
            payload = json.dumps({"global_step": step})
        elif name == "adapter_config.json":
            payload = "{}"
        else:
            payload = name
        (checkpoint / name).write_text(payload, encoding="utf-8")
    return checkpoint


class CheckpointResolutionTest(unittest.TestCase):
    def test_empty_output_starts_from_base(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertIsNone(resolve_resume_checkpoint(Path(temp_dir)))

    def test_selects_the_highest_complete_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_checkpoint(root, 64)
            latest = write_checkpoint(root, 256)
            result = resolve_resume_checkpoint(root)
            self.assertEqual(result["path"], str(latest.resolve()))
            self.assertEqual(result["step"], 256)

    def test_rejects_a_corrupt_latest_checkpoint_without_falling_back(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            write_checkpoint(root, 64)
            write_checkpoint(root, 256, omit="optimizer.pt")
            with self.assertRaisesRegex(RuntimeError, "optimizer.pt"):
                resolve_resume_checkpoint(root)

    def test_rejects_non_checkpoint_output(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "adapter_config.json").write_text("{}", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "non-empty"):
                resolve_resume_checkpoint(root)

    def test_bound_output_with_only_root_binding_can_restart_before_first_save(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output"
            log_root = root / "logs"
            identity = bound_identity(output)
            write_run_binding(output, identity)
            self.assertIsNone(
                resolve_resume_checkpoint(
                    output,
                    expected_run_identity=identity,
                    log_root=log_root,
                )
            )

    def test_bound_checkpoint_requires_manifest_and_file_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output"
            log_root = root / "logs"
            run_dir = log_root / "run-1"
            run_dir.mkdir(parents=True)
            identity = bound_identity(output)
            write_run_binding(output, identity)
            manifest = {
                "status": "running",
                "profile": identity["profile"],
                "stage": identity["stage"],
                "resolved_config": identity["resolved_config"],
                "input_integrity": bound_integrity(),
                "implementation_fingerprint": identity["implementation_fingerprint"],
                "run_identity": identity,
            }
            origin = run_dir / "manifest.json"
            origin.write_text(json.dumps(manifest), encoding="utf-8")
            checkpoint = write_checkpoint(output, 8)
            write_checkpoint_binding(
                checkpoint, run_identity=identity, origin_manifest=origin
            )
            result = resolve_resume_checkpoint(
                output,
                expected_run_identity=identity,
                log_root=log_root,
            )
            self.assertEqual(result["step"], 8)
            (checkpoint / "optimizer.pt").write_text("tampered", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "hashes"):
                resolve_resume_checkpoint(
                    output,
                    expected_run_identity=identity,
                    log_root=log_root,
                )

    def test_bound_resume_rejects_checkpoint_without_binding(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output"
            identity = bound_identity(output)
            write_run_binding(output, identity)
            write_checkpoint(output, 8)
            with self.assertRaisesRegex(
                RuntimeError, CHECKPOINT_BINDING_FILE.replace(".", "\\.")
            ):
                resolve_resume_checkpoint(
                    output,
                    expected_run_identity=identity,
                    log_root=root / "logs",
                )

    def test_bound_resume_rejects_a_manifest_from_another_profile(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output"
            log_root = root / "logs"
            run_dir = log_root / "run-1"
            run_dir.mkdir(parents=True)
            identity = bound_identity(output)
            write_run_binding(output, identity)
            manifest = {
                "status": "failed",
                "profile": "baseline",
                "stage": identity["stage"],
                "resolved_config": identity["resolved_config"],
                "input_integrity": bound_integrity(),
                "implementation_fingerprint": identity["implementation_fingerprint"],
                "run_identity": identity,
            }
            origin = run_dir / "manifest.json"
            origin.write_text(json.dumps(manifest), encoding="utf-8")
            checkpoint = write_checkpoint(output, 8)
            write_checkpoint_binding(
                checkpoint, run_identity=identity, origin_manifest=origin
            )
            with self.assertRaisesRegex(RuntimeError, "profile"):
                resolve_resume_checkpoint(
                    output,
                    expected_run_identity=identity,
                    log_root=log_root,
                )

    def test_trainer_callback_binds_a_completed_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output = root / "output"
            origin = root / "logs/run/manifest.json"
            origin.parent.mkdir(parents=True)
            origin.write_text("{}", encoding="utf-8")
            output.mkdir()
            checkpoint = write_checkpoint(output, 8)
            callback = CheckpointBindingCallback(
                run_identity=bound_identity(output),
                origin_manifest=origin,
            )
            callback.on_save(
                SimpleNamespace(output_dir=str(output), should_save=True),
                SimpleNamespace(global_step=8),
                SimpleNamespace(),
            )
            self.assertTrue((checkpoint / CHECKPOINT_BINDING_FILE).is_file())


def bound_integrity() -> dict:
    return {
        "profile": "frontier_feedbackcore_listwise_invariant_v1",
        "lock_path": "/lock",
        "lock_sha256": "lock",
        "model_root": "/model",
        "model_files": {},
        "source_dataset": {},
        "derived_dataset": {},
        "dataset_info": {},
        "llamafactory": {},
    }


def bound_identity(output: Path) -> dict:
    return {
        "schema_version": 1,
        "profile": "frontier_feedbackcore_listwise_invariant_v1",
        "stage": "frontier_full_epoch_001",
        "output_dir": str(output.resolve()),
        "input_artifact_identity": artifact_identity(bound_integrity()),
        "resolved_config": {"output_dir": str(output.resolve())},
        "custom_loss": {"gamma": 2.0, "item_weight": 3.0, "chunk_size": 512},
        "implementation_fingerprint": {"sha256": "impl", "files": {}},
    }


if __name__ == "__main__":
    unittest.main()
