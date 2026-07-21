from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ksllm4rec_rloo.cli import build_parser
from ksllm4rec_rloo.config import approved_config
from ksllm4rec_rloo.probe import probe_run_signature


class ProbeBindingTest(unittest.TestCase):
    def test_signature_embeds_the_training_runtime_signature(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            adapter = root / "adapter"
            trie = root / "trie"
            adapter.mkdir()
            trie.mkdir()
            marker = {"sha256": "a" * 64, "inputs": {"runtime": "bound"}}
            with (
                patch(
                    "ksllm4rec_rloo.probe.runtime_signature",
                    return_value=marker,
                ) as runtime,
                patch(
                    "ksllm4rec_rloo.probe.file_record",
                    side_effect=lambda path: {"path": str(path), "sha256": "b" * 64},
                ),
                patch(
                    "ksllm4rec_rloo.probe.require_file",
                    side_effect=lambda path, expected: {
                        "path": str(path),
                        "sha256": expected,
                    },
                ),
            ):
                value = probe_run_signature(
                    approved_config(),
                    adapter,
                    trie,
                    groups_path=root / "groups.jsonl",
                    calibration_ids_path=root / "calibration.json",
                    config_path=root / "config.yaml",
                )
        self.assertEqual(value["inputs"]["training_runtime_signature"], marker)
        runtime.assert_called_once()

    def test_cli_requires_groups_and_calibration_for_probe(self) -> None:
        parser = build_parser()
        args = parser.parse_args(
            [
                "probe",
                "--groups",
                "groups.jsonl",
                "--trie-dir",
                "trie",
                "--calibration-ids",
                "calibration.json",
                "--policy-adapter",
                "adapter",
                "--output-dir",
                "probe",
            ]
        )
        self.assertEqual(args.groups, Path("groups.jsonl"))
        self.assertEqual(args.calibration_ids, Path("calibration.json"))


if __name__ == "__main__":
    unittest.main()
