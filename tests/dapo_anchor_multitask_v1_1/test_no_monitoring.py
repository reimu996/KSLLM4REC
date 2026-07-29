from __future__ import annotations

import unittest
from pathlib import Path


_ROOT = Path(__file__).resolve().parents[2]
_PACKAGE = _ROOT / "src/ksllm4rec_dapo_anchor_multitask_v1_1"
_SCRIPTS = (
    _ROOT
    / "scripts/dapo_anchor/frontier_sft372_dapo_anchor_multitask_v1_1_e2"
)


class NoTrainingMonitoringTest(unittest.TestCase):
    def test_formal_training_has_no_health_or_peak_memory_monitoring(self) -> None:
        trainer = (_PACKAGE / "trainer.py").read_text(encoding="utf-8")
        formal_training = trainer[trainer.index("def _run_training_impl(") :]
        formal_training = formal_training[: formal_training.index("def run_training(")]

        for forbidden in (
            "health.jsonl",
            "reset_peak_memory_stats",
            "max_memory_reserved",
            "peak_reserved_gib",
        ):
            self.assertNotIn(forbidden, formal_training)

    def test_status_and_persistent_stdout_entrypoints_are_absent(self) -> None:
        self.assertFalse((_SCRIPTS / "status.sh").exists())
        common = (_SCRIPTS / "common.sh").read_text(encoding="utf-8")
        run_full = (_SCRIPTS / "run_full.sh").read_text(encoding="utf-8")
        readme = (_SCRIPTS / "README.md").read_text(encoding="utf-8")

        self.assertNotIn("TRAIN_LOG", common)
        self.assertNotIn("full_train.stdout.log", common + run_full + readme)
        self.assertNotIn("tee -a", run_full)
        self.assertNotIn("status.sh", readme)

    def test_offline_pilot_retains_reserved_memory_gate(self) -> None:
        trainer = (_PACKAGE / "trainer.py").read_text(encoding="utf-8")
        pilot = trainer[trainer.index("def run_one_window_pilot(") :]

        self.assertIn("reset_peak_memory_stats", pilot)
        self.assertIn("max_memory_reserved", pilot)
        self.assertIn('config["memory"]["max_reserved_gib"]', pilot)

    def test_formal_api_rejects_recovery_audit_and_dense_test_hooks(self) -> None:
        from ksllm4rec_dapo_anchor_multitask_v1_1.trainer import run_training

        with self.assertRaisesRegex(ValueError, "source exhaustion"):
            run_training(
                {},
                {},
                output_dir=Path("/tmp/not-used"),
                formal=True,
                stop_after_source_blocks=1,
            )
        with self.assertRaisesRegex(ValueError, "dense-scoring"):
            run_training(
                {},
                {},
                output_dir=Path("/tmp/not-used"),
                formal=True,
                dense_scoring=True,
            )


if __name__ == "__main__":
    unittest.main()
