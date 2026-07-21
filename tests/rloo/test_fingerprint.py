from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ksllm4rec_rloo.fingerprint import (
    discover_runtime_code_files,
    frozen_model_inputs,
    runtime_signature,
    validate_runtime_signature,
)


class RuntimeFingerprintTest(unittest.TestCase):
    def make_inputs(self, root: Path) -> tuple[dict, dict[str, Path]]:
        base = root / "base"
        adapter = root / "adapter-r64"
        tokenizer = root / "tokenizer"
        trie = root / "trie"
        package = root / "src" / "ksllm4rec_rloo"
        scripts = root / "scripts" / "rloo"
        for directory in (base, adapter, tokenizer, trie, package, scripts):
            directory.mkdir(parents=True)
        (base / "config.json").write_text("{}", encoding="utf-8")
        (base / "model.safetensors").write_bytes(b"base")
        (adapter / "adapter_model.safetensors").write_bytes(b"adapter-r64")
        (adapter / "adapter_config.json").write_text(
            '{"r":64,"lora_alpha":64}', encoding="utf-8"
        )
        (tokenizer / "tokenizer.json").write_text("{}", encoding="utf-8")
        (tokenizer / "tokenizer_config.json").write_text("{}", encoding="utf-8")
        groups = root / "groups.jsonl"
        groups.write_text('{"group_id":"g0"}\n', encoding="utf-8")
        (trie / "manifest.json").write_text('{"leaves":905469}', encoding="utf-8")
        (trie / "a_values.npy").write_bytes(b"trie-array")
        calibration = root / "calibration_ids.json"
        calibration.write_text('["g0"]\n', encoding="utf-8")
        config_file = root / "config.yaml"
        config_file.write_text("spec_version: '2.0'\n", encoding="utf-8")
        runtime = package / "trainer.py"
        runtime.write_text("VALUE = 1\n", encoding="utf-8")
        script = scripts / "run.sh"
        script.write_text("#!/bin/sh\n", encoding="utf-8")
        config = {
            "schema_version": 1,
            "spec_version": "2.0",
            "model": {
                "base_model": str(base),
                "sft_adapter": str(adapter),
                "tokenizer": str(tokenizer),
            },
            "reward": {"exact": 1.0, "same_ab": 0.4, "same_a": 0.15},
        }
        return config, {
            "base": base,
            "adapter": adapter,
            "tokenizer": tokenizer,
            "groups": groups,
            "trie": trie,
            "calibration": calibration,
            "config_file": config_file,
            "runtime": runtime,
            "script": script,
        }

    def get_signature(self, root: Path, config: dict, paths: dict[str, Path]) -> dict:
        with patch(
            "ksllm4rec_rloo.fingerprint.software_versions",
            return_value={"python": "test", "torch": "test"},
        ):
            return runtime_signature(
                config,
                paths["groups"],
                paths["trie"],
                paths["calibration"],
                config_path=paths["config_file"],
                project_root=root,
            )

    def test_signature_binds_config_models_data_calibration_and_all_code(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, paths = self.make_inputs(root)
            before = self.get_signature(root, config, paths)
            validate_runtime_signature(before)
            cases = (
                ("config", paths["config_file"], b"spec_version: 'changed'\n"),
                ("base", paths["base"] / "model.safetensors", b"base-changed"),
                (
                    "adapter",
                    paths["adapter"] / "adapter_model.safetensors",
                    b"adapter-changed",
                ),
                ("tokenizer", paths["tokenizer"] / "tokenizer.json", b"tokenizer"),
                ("groups", paths["groups"], b'{"group_id":"g1"}\n'),
                ("trie", paths["trie"] / "a_values.npy", b"trie-changed"),
                ("calibration", paths["calibration"], b'["g1"]\n'),
                ("runtime", paths["runtime"], b"VALUE = 2\n"),
                ("script", paths["script"], b"#!/bin/sh\ntrue\n"),
            )
            for label, path, replacement in cases:
                original = path.read_bytes()
                path.write_bytes(replacement)
                after = self.get_signature(root, config, paths)
                self.assertNotEqual(
                    before["sha256"], after["sha256"], msg=f"unbound input: {label}"
                )
                path.write_bytes(original)

            changed_config = dict(config)
            changed_config["spec_version"] = "2.1"
            self.assertNotEqual(
                before["sha256"],
                self.get_signature(root, changed_config, paths)["sha256"],
            )

    def test_actual_adapter_hash_is_used_without_historical_profile(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, paths = self.make_inputs(root)
            records = frozen_model_inputs(config)
            adapter_record = records["sft_adapter"]["files"]
            self.assertEqual(
                set(adapter_record),
                {"adapter_config.json", "adapter_model.safetensors"},
            )
            self.assertEqual(
                records["sft_adapter"]["path"], str(paths["adapter"].resolve())
            )

    def test_runtime_discovery_covers_package_and_scripts_recursively(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, paths = self.make_inputs(root)
            nested = root / "src" / "ksllm4rec_rloo" / "nested" / "module.py"
            nested.parent.mkdir()
            nested.write_text("VALUE = 3\n", encoding="utf-8")
            dependencies = []
            for package, filename in (
                ("ksllm4rec_grpo", "constraint.py"),
                ("ksllm4rec_orpo", "data.py"),
                ("ksllm4rec_sft", "data.py"),
            ):
                dependency = root / "src" / package / filename
                dependency.parent.mkdir(parents=True)
                dependency.write_text("VALUE = 1\n", encoding="utf-8")
                dependencies.append(dependency.resolve())
            discovered = set(discover_runtime_code_files(root))
            self.assertIn(paths["runtime"].resolve(), discovered)
            self.assertIn(paths["script"].resolve(), discovered)
            self.assertIn(nested.resolve(), discovered)
            for dependency in dependencies:
                self.assertIn(dependency, discovered)

    def test_tampered_signature_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config, paths = self.make_inputs(root)
            value = self.get_signature(root, config, paths)
            value["inputs"]["schema_version"] = 999
            with self.assertRaisesRegex(RuntimeError, "self-inconsistent"):
                validate_runtime_signature(value)


if __name__ == "__main__":
    unittest.main()
