from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ksllm4rec_sft.integrity import (
    snapshot_output_artifacts,
    verify_artifact_lock,
    verify_environment_lock,
    verify_output_artifacts,
)


def locked_file(path: Path) -> dict[str, object]:
    payload = path.read_bytes()
    return {
        "path": str(path),
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


class ArtifactIntegrityTest(unittest.TestCase):
    def test_verifies_all_locked_files_and_commit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model"
            model.mkdir()
            model_file = model / "config.json"
            model_file.write_text("{}", encoding="utf-8")
            source = root / "source.jsonl"
            derived = root / "derived.jsonl"
            source.write_text("source", encoding="utf-8")
            derived.write_text("derived", encoding="utf-8")
            commit = "a" * 40
            tree = "b" * 40
            lock = {
                "schema_version": 1,
                "llamafactory": {
                    "path": str(root),
                    "commit": commit,
                    "tree": tree,
                },
                "model": {
                    "path": str(model),
                    "files": {"config.json": locked_file(model_file)},
                },
                "source_dataset": {**locked_file(source), "records": 1},
                "derived_dataset": {**locked_file(derived), "records": 1},
            }
            lock_path = root / "artifacts.lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            identity = subprocess.CompletedProcess(
                [], 0, stdout=f"{commit}\n{tree}\n", stderr=""
            )
            clean = subprocess.CompletedProcess([], 0, stdout="", stderr="")
            with patch(
                "ksllm4rec_sft.integrity.subprocess.run",
                side_effect=[identity, clean],
            ):
                report = verify_artifact_lock(lock_path)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["llamafactory"]["commit"], commit)
            self.assertTrue(report["llamafactory"]["working_tree_clean"])

    def test_rejects_dirty_llamafactory_working_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model"
            model.mkdir()
            model_file = model / "config.json"
            model_file.write_text("{}", encoding="utf-8")
            source = root / "source"
            derived = root / "derived"
            source.write_text("s", encoding="utf-8")
            derived.write_text("d", encoding="utf-8")
            commit = "a" * 40
            tree = "b" * 40
            lock = {
                "schema_version": 1,
                "llamafactory": {
                    "path": str(root),
                    "commit": commit,
                    "tree": tree,
                },
                "model": {
                    "path": str(model),
                    "files": {"config.json": locked_file(model_file)},
                },
                "source_dataset": locked_file(source),
                "derived_dataset": locked_file(derived),
            }
            lock_path = root / "lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            identity = subprocess.CompletedProcess(
                [], 0, stdout=f"{commit}\n{tree}\n", stderr=""
            )
            dirty = subprocess.CompletedProcess(
                [], 0, stdout=" M src/llamafactory/data/foo.py\n", stderr=""
            )
            with (
                patch(
                    "ksllm4rec_sft.integrity.subprocess.run",
                    side_effect=[identity, dirty],
                ),
                self.assertRaisesRegex(RuntimeError, "not clean"),
            ):
                verify_artifact_lock(lock_path)

    def test_rejects_content_drift_with_unchanged_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "model"
            model.mkdir()
            model_file = model / "config.json"
            model_file.write_text("ab", encoding="utf-8")
            source = root / "source"
            derived = root / "derived"
            source.write_text("s", encoding="utf-8")
            derived.write_text("d", encoding="utf-8")
            lock = {
                "schema_version": 1,
                "llamafactory": {"path": str(root), "commit": "a" * 40},
                "model": {
                    "path": str(model),
                    "files": {"config.json": locked_file(model_file)},
                },
                "source_dataset": locked_file(source),
                "derived_dataset": locked_file(derived),
            }
            lock_path = root / "lock.json"
            lock_path.write_text(json.dumps(lock), encoding="utf-8")
            model_file.write_text("cd", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA256 mismatch"):
                verify_artifact_lock(lock_path)

    def test_environment_lock_matches_pip_freeze(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "environment.lock.txt"
            lock_path.write_text("package==1.0\n", encoding="utf-8")
            completed = subprocess.CompletedProcess(
                [], 0, stdout="package==1.0\n", stderr=""
            )
            with patch(
                "ksllm4rec_sft.integrity.subprocess.run", return_value=completed
            ):
                report = verify_environment_lock(lock_path)
            self.assertEqual(report["status"], "passed")
            self.assertEqual(report["packages"], 1)

    def test_environment_lock_rejects_version_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "environment.lock.txt"
            lock_path.write_text("package==1.0\n", encoding="utf-8")
            completed = subprocess.CompletedProcess(
                [], 0, stdout="package==2.0\n", stderr=""
            )
            with (
                patch(
                    "ksllm4rec_sft.integrity.subprocess.run",
                    return_value=completed,
                ),
                self.assertRaisesRegex(RuntimeError, "package==2.0"),
            ):
                verify_environment_lock(lock_path)

    def test_environment_lock_rejects_python_patch_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            lock_path = Path(temp_dir) / "environment.lock.txt"
            lock_path.write_text("package==1.0\n", encoding="utf-8")
            with (
                patch(
                    "ksllm4rec_sft.integrity.platform.python_version",
                    return_value="3.11.14",
                ),
                self.assertRaisesRegex(RuntimeError, "3.11.15"),
            ):
                verify_environment_lock(lock_path)

    def test_output_snapshot_binds_all_files_by_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            for name in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "trainer_state.json",
                "train_results.json",
            ):
                (root / name).write_text(name, encoding="utf-8")
            snapshot = snapshot_output_artifacts(root)
            verified = verify_output_artifacts(root, snapshot)
            self.assertEqual(set(verified), set(snapshot))
            (root / "adapter_model.safetensors").write_text(
                "changed content", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                verify_output_artifacts(root, snapshot)


if __name__ == "__main__":
    unittest.main()
