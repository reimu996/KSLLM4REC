from __future__ import annotations

import unittest

from ksllm4rec_rloo_dapo.cli import build_parser


class CLITest(unittest.TestCase):
    def test_formal_train_requires_every_gate_report(self) -> None:
        parser = build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args(["train", "--config", "config.yaml"])

    def test_no_anchor_or_reference_commands_exist(self) -> None:
        help_text = build_parser().format_help().lower()
        self.assertNotIn("anchor", help_text)
        self.assertNotIn("reference", help_text)


if __name__ == "__main__":
    unittest.main()
