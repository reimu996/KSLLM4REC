from __future__ import annotations

import argparse
import unittest
from pathlib import Path
from unittest.mock import patch

from ksllm4rec_sft.cli import _resolve_profile_resume, build_parser
from ksllm4rec_sft.profiles import (
    BASELINE_PROFILE,
    FRONTIER_PROFILE,
    FRONTIER_PROFILE_NAME,
    PROFILE_NAMES,
    get_profile,
)


class SFTProfileTest(unittest.TestCase):
    def test_frontier_identity_is_the_approved_immutable_source(self) -> None:
        self.assertEqual(FRONTIER_PROFILE.source_records, 63_700)
        self.assertEqual(FRONTIER_PROFILE.source_size, 269_105_772)
        self.assertEqual(
            FRONTIER_PROFILE.source_sha256,
            "9e3465cedbdaf784ab4720699649c76c256414e2c2089efcd7657f63bae7449a",
        )
        self.assertEqual(
            str(FRONTIER_PROFILE.source_path),
            "/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/train.jsonl",
        )

    def test_profiles_have_disjoint_dataset_and_stage_identities(self) -> None:
        self.assertNotEqual(
            BASELINE_PROFILE.dataset_name, FRONTIER_PROFILE.dataset_name
        )
        self.assertNotEqual(BASELINE_PROFILE.dataset_dir, FRONTIER_PROFILE.dataset_dir)
        self.assertNotEqual(BASELINE_PROFILE.full_stage, FRONTIER_PROFILE.full_stage)
        self.assertTrue(
            set(BASELINE_PROFILE.gate_cutoffs).isdisjoint(FRONTIER_PROFILE.gate_cutoffs)
        )

    def test_unknown_profile_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown SFT profile"):
            get_profile("unknown")

    def test_every_pipeline_command_accepts_profile(self) -> None:
        parser = build_parser()
        subparsers = next(
            action
            for action in parser._actions
            if isinstance(action, argparse._SubParsersAction)
        )
        expected_commands = {
            "prepare-data",
            "create-lock",
            "preflight",
            "train",
            "check-gates",
            "config-check",
            "verify",
        }
        self.assertTrue(expected_commands.issubset(subparsers.choices))
        for name in expected_commands:
            option_strings = {
                option
                for action in subparsers.choices[name]._actions
                for option in action.option_strings
            }
            self.assertIn("--profile", option_strings, name)

    def test_cli_defaults_to_baseline_and_accepts_frontier(self) -> None:
        parser = build_parser()
        baseline = parser.parse_args(
            ["prepare-data", "--source", "/tmp/s", "--output-dir", "/tmp/o"]
        )
        self.assertEqual(baseline.profile, BASELINE_PROFILE.name)
        frontier = parser.parse_args(
            [
                "prepare-data",
                "--profile",
                FRONTIER_PROFILE_NAME,
                "--source",
                "/tmp/s",
                "--output-dir",
                "/tmp/o",
            ]
        )
        self.assertEqual(frontier.profile, FRONTIER_PROFILE.name)
        self.assertEqual(PROFILE_NAMES, (BASELINE_PROFILE.name, FRONTIER_PROFILE.name))

    def test_resume_resolution_runs_only_for_the_selected_profile_full_stage(
        self,
    ) -> None:
        with patch(
            "ksllm4rec_sft.checkpoint.resolve_resume_checkpoint",
            return_value={"path": "/checkpoint"},
        ) as resolver:
            self.assertIsNone(
                _resolve_profile_resume(
                    "full_epoch_001", FRONTIER_PROFILE, Path("/output")
                )
            )
            resolver.assert_not_called()
            self.assertEqual(
                _resolve_profile_resume(
                    FRONTIER_PROFILE.full_stage,
                    FRONTIER_PROFILE,
                    Path("/output"),
                ),
                {"path": "/checkpoint"},
            )
            resolver.assert_called_once_with(Path("/output"))


if __name__ == "__main__":
    unittest.main()
