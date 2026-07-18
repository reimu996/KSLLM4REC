from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from ksllm4rec_sft.integrity import (
    artifact_identity,
    create_profile_artifact_lock,
    snapshot_output_artifacts,
    verify_artifact_lock,
    verify_environment_lock,
    verify_output_artifacts,
)
from ksllm4rec_sft.data import expected_dataset_info
from ksllm4rec_sft.profiles import BASELINE_PROFILE, FRONTIER_PROFILE, MODEL_PATH


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
            self.assertIsNone(artifact_identity(report)["dataset_info"]["path"])

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

    def test_profile_lock_preserves_each_record_and_uses_fixed_source_identity(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "source.jsonl"
            derived_dir = root / "derived"
            derived_dir.mkdir()
            derived = derived_dir / "train_alpaca.jsonl"
            source.write_text(
                '[{"system":"s","prompt":"p","response":"r"}]\n',
                encoding="utf-8",
            )
            derived.write_text(
                '{"instruction":"p","input":"","output":"r","system":"s"}\n',
                encoding="utf-8",
            )
            (derived_dir / "dataset_info.json").write_text(
                json.dumps(expected_dataset_info("frontier_dataset", derived.name)),
                encoding="utf-8",
            )
            base_lock = root / "base.lock.json"
            output_lock = root / "frontier.lock.json"
            base_lock.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "llamafactory": {"path": "/lf", "commit": "c", "tree": "t"},
                        "model": {"path": str(MODEL_PATH), "files": {}},
                    }
                ),
                encoding="utf-8",
            )
            selected = replace(
                FRONTIER_PROFILE,
                source_path=source,
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                source_size=source.stat().st_size,
                source_records=1,
                dataset_dir=derived_dir,
                artifact_lock_path=output_lock,
                dataset_name="frontier_dataset",
            )
            base_source = root / "base-source"
            base_source.write_text("base", encoding="utf-8")
            base_derived_dir = root / "base-derived"
            base_derived_dir.mkdir()
            base_derived = base_derived_dir / "train_alpaca.jsonl"
            base_derived.write_text("base-derived", encoding="utf-8")
            baseline = replace(
                BASELINE_PROFILE,
                artifact_lock_path=base_lock,
                source_path=base_source,
                source_sha256=hashlib.sha256(base_source.read_bytes()).hexdigest(),
                source_size=base_source.stat().st_size,
                source_records=1,
                dataset_dir=base_derived_dir,
            )
            base_report = {
                "status": "passed",
                "profile": None,
                "lock_path": str(base_lock.resolve()),
                "lock_sha256": "base-lock-sha",
                "model_root": str(MODEL_PATH.resolve()),
                "model_files": {},
                "source_dataset": {
                    "path": str(base_source.resolve()),
                    "size": base_source.stat().st_size,
                    "sha256": baseline.source_sha256,
                    "records": 1,
                },
                "derived_dataset": {
                    "path": str(base_derived.resolve()),
                    "size": base_derived.stat().st_size,
                    "sha256": hashlib.sha256(base_derived.read_bytes()).hexdigest(),
                    "records": 1,
                },
                "llamafactory": {"path": "/lf", "commit": "c", "tree": "t"},
            }

            def select_profile(value):
                return baseline if value == "baseline" else value

            with (
                patch(
                    "ksllm4rec_sft.integrity.get_profile",
                    side_effect=select_profile,
                ),
                patch(
                    "ksllm4rec_sft.integrity.verify_artifact_lock",
                    return_value=base_report,
                ),
            ):
                report = create_profile_artifact_lock(
                    profile=selected,
                    source_path=source,
                    derived_path=derived,
                    base_lock_path=base_lock,
                    output_path=output_lock,
                )
            created = json.loads(output_lock.read_text(encoding="utf-8"))
            self.assertEqual(report["status"], "passed")
            self.assertEqual(created["profile"], selected.name)
            self.assertEqual(created["source_dataset"]["records"], 1)
            self.assertEqual(
                created["derived_dataset"]["sha256"],
                hashlib.sha256(derived.read_bytes()).hexdigest(),
            )

    def test_profile_lock_rejects_an_unapproved_source_path(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            with self.assertRaisesRegex(RuntimeError, "Source path"):
                create_profile_artifact_lock(
                    profile=FRONTIER_PROFILE,
                    source_path=root / "different-source.jsonl",
                    derived_path=FRONTIER_PROFILE.derived_path,
                    base_lock_path=BASELINE_PROFILE.artifact_lock_path,
                    output_path=FRONTIER_PROFILE.artifact_lock_path,
                )


if __name__ == "__main__":
    unittest.main()
