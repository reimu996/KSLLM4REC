from __future__ import annotations

import argparse
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ksllm4rec_rloo_dapo import contract
from ksllm4rec_rloo_dapo.cli import (
    _load,
    _require_execution_device,
    _require_gate_output,
    _require_pilot_dir,
    build_parser,
    train_command,
)
from ksllm4rec_rloo_dapo.config import approved_config


class CLITest(unittest.TestCase):
    def test_formal_train_requires_every_gate_report(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["train", "--config", "config.yaml"])

    def test_no_anchor_or_reference_commands_exist(self) -> None:
        help_text = build_parser().format_help().lower()
        self.assertNotIn("anchor", help_text)
        self.assertNotIn("reference", help_text)

    def test_cpu_report_argument_remains_optional_for_the_legacy_profile(self) -> None:
        arguments = [
            "train",
            "--config",
            "config.yaml",
            "--structure-report",
            "structure.json",
            "--probability-report",
            "probability.json",
            "--memory-report",
            "memory.json",
            "--throughput-report",
            "throughput.json",
            "--pilot-report",
            "pilot.json",
            "--output-dir",
            "run",
        ]
        parser = build_parser()
        legacy = parser.parse_args(arguments)
        self.assertIsNone(legacy.cpu_report)
        parsed = parser.parse_args([*arguments, "--cpu-report", "cpu.json"])
        self.assertEqual(parsed.cpu_report, Path("cpu.json"))

    def test_sft372_handler_rejects_a_missing_cpu_report_before_gpu_access(
        self,
    ) -> None:
        config = approved_config(contract.SFT372_PROFILE)
        profile = contract.frozen_profile(contract.SFT372_PROFILE)
        args = argparse.Namespace(
            device="cuda:0",
            output_dir=profile.run_dir,
            cpu_report=None,
        )
        with (
            patch(
                "ksllm4rec_rloo_dapo.cli._load",
                return_value=(config, {"sha256": "unused"}),
            ),
            patch("ksllm4rec_rloo_dapo.cli.require_gpu_identity") as gpu,
            self.assertRaisesRegex(RuntimeError, "requires a CPU gate report"),
        ):
            train_command(args)
        gpu.assert_not_called()

    def test_visible_gpu_set_is_frozen_only_for_the_sft372_profile(self) -> None:
        with patch.dict("os.environ", {}, clear=True):
            _require_execution_device("cuda:0", require_visibility=False)
            with self.assertRaisesRegex(RuntimeError, "CUDA_VISIBLE_DEVICES"):
                _require_execution_device("cuda:0", require_visibility=True)
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": "0"}, clear=True):
            _require_execution_device("cuda:0", require_visibility=True)

    def test_load_rejects_valid_config_content_from_non_frozen_path(self) -> None:
        profile = contract.frozen_profile(contract.SFT372_PROFILE)
        with tempfile.TemporaryDirectory() as temporary:
            copied = Path(temporary) / profile.config_path.name
            copied.write_bytes(profile.config_path.read_bytes())
            with self.assertRaisesRegex(RuntimeError, "frozen profile path"):
                _load(argparse.Namespace(config=copied))

    def test_legacy_profile_keeps_config_and_artifact_overrides(self) -> None:
        profile = contract.frozen_profile(contract.PROFILE)
        config = approved_config(contract.PROFILE)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            copied = root / profile.config_path.name
            copied.write_bytes(profile.config_path.read_bytes())
            with patch(
                "ksllm4rec_rloo_dapo.cli.runtime_signature",
                return_value={"sha256": "legacy"},
            ):
                loaded, signature = _load(argparse.Namespace(config=copied))
            self.assertEqual(loaded, config)
            self.assertEqual(signature, {"sha256": "legacy"})
            _require_gate_output(
                config, root / "gates" / "structure.json", "structure.json"
            )
            _require_pilot_dir(config, root / "pilot")

    def test_gate_output_accepts_only_fixed_or_well_formed_staging_path(self) -> None:
        config = approved_config(contract.SFT372_PROFILE)
        profile = contract.frozen_profile(contract.SFT372_PROFILE)
        fixed = profile.log_dir / "gates" / "structure.json"
        staged = (
            profile.log_dir
            / ".gates.staging.20260724T120102Z.12345"
            / "reports"
            / "structure.json"
        )
        _require_gate_output(config, fixed, "structure.json")
        _require_gate_output(config, staged, "structure.json")
        invalid = (
            profile.log_dir / ".gates.staging." / "reports" / "structure.json",
            profile.log_dir
            / ".gates.staging.not-a-run-id"
            / "reports"
            / "structure.json",
            profile.log_dir
            / ".gates.staging.20260724T120102Z.12345"
            / "other"
            / "structure.json",
            profile.log_dir
            / ".gates.staging.20260724T120102Z.12345"
            / "reports"
            / "memory.json",
            profile.log_dir.parent / "outside" / "structure.json",
        )
        for path in invalid:
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                _require_gate_output(config, path, "structure.json")

    def test_pilot_accepts_only_fixed_or_well_formed_staging_path(self) -> None:
        config = approved_config(contract.SFT372_PROFILE)
        profile = contract.frozen_profile(contract.SFT372_PROFILE)
        staged = profile.log_dir / ".gates.staging.20260724T120102Z.12345" / "pilot"
        _require_pilot_dir(config, profile.pilot_dir)
        _require_pilot_dir(config, staged)
        invalid = (
            profile.log_dir / ".gates.staging." / "pilot",
            profile.log_dir / ".gates.staging.not-a-run-id" / "pilot",
            profile.log_dir / ".gates.staging.20260724T120102Z.12345" / "not-pilot",
            profile.log_dir.parent / "outside" / "pilot",
        )
        for path in invalid:
            with self.subTest(path=path), self.assertRaises(RuntimeError):
                _require_pilot_dir(config, path)


if __name__ == "__main__":
    unittest.main()
