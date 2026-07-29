from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ksllm4rec_dapo_anchor_multitask_v1_1._infra.dapo_fingerprint import (
    discover_runtime_code_files,
    runtime_code_fingerprint,
    runtime_signature,
    validate_runtime_signature,
)


class MultitaskFingerprintTest(unittest.TestCase):
    def make_signature_inputs(self, root: Path) -> tuple[dict, dict[str, Path]]:
        base = root / "base"
        adapter = root / "adapter"
        tokenizer = root / "tokenizer"
        recommendation = root / "recommendation"
        text = root / "text"
        trie = root / "trie"
        for directory in (base, adapter, tokenizer, recommendation, text, trie):
            directory.mkdir(parents=True)
        (base / "config.json").write_text("{}\n", encoding="utf-8")
        (base / "model.safetensors").write_bytes(b"base")
        (adapter / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter")
        (tokenizer / "tokenizer.json").write_text("{}\n", encoding="utf-8")
        (tokenizer / "tokenizer_config.json").write_text("{}\n", encoding="utf-8")
        for group_dir, marker in ((recommendation, "rec"), (text, "text")):
            (group_dir / "groups.jsonl").write_text(
                f'{{"group_id":"{marker}"}}\n', encoding="utf-8"
            )
            (group_dir / "data_manifest.json").write_text(
                f'{{"artifact":"{marker}"}}\n', encoding="utf-8"
            )
        (trie / "manifest.json").write_text('{"leaves":2}\n', encoding="utf-8")
        (trie / "values.npy").write_bytes(b"trie")
        paths = {
            "source": root / "train.jsonl",
            "provenance": root / "provenance.jsonl",
            "fixed_probe": root / "fixed_probe.jsonl",
            "config_file": root / "config.yaml",
            "recommendation": recommendation,
            "text": text,
            "trie": trie,
        }
        paths["source"].write_text("{}\n", encoding="utf-8")
        paths["provenance"].write_text("{}\n", encoding="utf-8")
        paths["fixed_probe"].write_text("{}\n", encoding="utf-8")
        paths["config_file"].write_text("spec: multitask-v1\n", encoding="utf-8")
        runtime = root / "src/ksllm4rec_dapo_anchor_multitask_v1_1/trainer.py"
        runtime.parent.mkdir(parents=True)
        runtime.write_text("VALUE = 1\n", encoding="utf-8")
        script = (
            root
            / "scripts/dapo_anchor/frontier_sft372_dapo_anchor_multitask_v1_1_e2/train.py"
        )
        script.parent.mkdir(parents=True)
        script.write_text("VALUE = 1\n", encoding="utf-8")
        status_script = script.parent / "status.sh"
        status_script.write_text("exit 0\n", encoding="utf-8")
        test_file = root / "tests/dapo_anchor_multitask_v1_1/test_trainer.py"
        test_file.parent.mkdir(parents=True)
        test_file.write_text("VALUE = 1\n", encoding="utf-8")
        paths.update(
            {
                "runtime": runtime,
                "script": script,
                "status_script": status_script,
                "test": test_file,
            }
        )
        config = {
            "schema_version": 4,
            "profile": "frontier_sft372_dapo_anchor_multitask_v1_1_e2",
            "model": {
                "base_model": str(base),
                "sft_adapter": str(adapter),
                "tokenizer": str(tokenizer),
            },
            "data": {
                "source": str(paths["source"]),
                "provenance": str(paths["provenance"]),
            },
            "evaluation": {"fixed_probe": str(paths["fixed_probe"])},
            "output": {
                "recommendation_groups_dir": str(recommendation),
                "text_to_sid_groups_dir": str(text),
                "trie_dir": str(trie),
            },
        }
        return config, paths

    def get_signature(
        self,
        root: Path,
        config: dict,
        paths: dict[str, Path],
    ) -> dict:
        with patch(
            "ksllm4rec_dapo_anchor_multitask_v1_1._infra.dapo_fingerprint.software_versions",
            return_value={"python": "test"},
        ):
            return runtime_signature(
                config,
                config_path=paths["config_file"],
                project_root=root,
                enforce_frozen_identity=False,
            )

    def test_discovery_scans_only_the_new_package_scripts_and_tests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            expected = {
                root / "src/ksllm4rec_dapo_anchor_multitask_v1_1/trainer.py",
                root
                / "scripts/dapo_anchor/frontier_sft372_dapo_anchor_multitask_v1_1_e2/train.py",
                root
                / "scripts/dapo_anchor/frontier_sft372_dapo_anchor_multitask_v1_1_e2/status.sh",
                root / "tests/dapo_anchor_multitask_v1_1/test_trainer.py",
            }
            forbidden = {
                root / "src/ksllm4rec_dapo_anchor_v2_2/trainer.py",
                root / "scripts/dapo_anchor/frontier_sft372_dapo_anchor_v2_2/train.py",
                root / "tests/dapo_anchor_v2_2/test_trainer.py",
            }
            for path in expected | forbidden:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("VALUE = 1\n", encoding="utf-8")

            discovered = set(discover_runtime_code_files(root))

            self.assertEqual(discovered, {path.resolve() for path in expected})
            self.assertTrue(discovered.isdisjoint(path.resolve() for path in forbidden))

    def test_explicit_v22_runtime_file_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            old = root / "src/ksllm4rec_dapo_anchor_v2_2/trainer.py"
            old.parent.mkdir(parents=True)
            old.write_text("VALUE = 1\n", encoding="utf-8")

            with self.assertRaisesRegex(RuntimeError, "outside Multitask V1"):
                runtime_code_fingerprint(root, [old])

    def test_signature_binds_both_group_files_and_both_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, paths = self.make_signature_inputs(root)
            with patch(
                "ksllm4rec_dapo_anchor_multitask_v1_1._infra.dapo_fingerprint.software_versions",
                return_value={"python": "test"},
            ):
                signature = runtime_signature(
                    config,
                    config_path=paths["config_file"],
                    project_root=root,
                    enforce_frozen_identity=False,
                )

            inputs = signature["inputs"]
            self.assertEqual(
                set(inputs["recommendation_groups"]), {"path", "groups", "manifest"}
            )
            self.assertEqual(
                set(inputs["text_to_sid_groups"]), {"path", "groups", "manifest"}
            )
            self.assertTrue(
                inputs["recommendation_groups"]["groups"]["path"].endswith(
                    "recommendation/groups.jsonl"
                )
            )
            self.assertTrue(
                inputs["text_to_sid_groups"]["manifest"]["path"].endswith(
                    "text/data_manifest.json"
                )
            )

    def test_production_mode_rejects_self_consistent_but_unfrozen_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, paths = self.make_signature_inputs(root)

            with patch(
                "ksllm4rec_dapo_anchor_multitask_v1_1._infra.dapo_fingerprint.software_versions",
                return_value={"python": "test"},
            ), self.assertRaisesRegex(RuntimeError, "source fingerprint mismatch"):
                runtime_signature(
                    config,
                    config_path=paths["config_file"],
                    project_root=root,
                )

    def test_signature_changes_when_any_required_input_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, paths = self.make_signature_inputs(root)
            before = self.get_signature(root, config, paths)
            validate_runtime_signature(before)
            cases = (
                paths["config_file"],
                paths["source"],
                paths["provenance"],
                Path(config["model"]["base_model"]) / "model.safetensors",
                Path(config["model"]["sft_adapter"])
                / "adapter_model.safetensors",
                paths["recommendation"] / "groups.jsonl",
                paths["recommendation"] / "data_manifest.json",
                paths["text"] / "groups.jsonl",
                paths["text"] / "data_manifest.json",
                paths["trie"] / "values.npy",
                paths["fixed_probe"],
                paths["runtime"],
                paths["script"],
                paths["status_script"],
                paths["test"],
            )
            for path in cases:
                original = path.read_bytes()
                path.write_bytes(original + b"changed")
                after = self.get_signature(root, config, paths)
                self.assertNotEqual(before["sha256"], after["sha256"], msg=str(path))
                path.write_bytes(original)

            changed_config = dict(config)
            changed_config["schema_version"] = 5
            after = self.get_signature(root, changed_config, paths)
            self.assertNotEqual(before["sha256"], after["sha256"])

if __name__ == "__main__":
    unittest.main()
