from __future__ import annotations

import unittest
from pathlib import Path

from ksllm4rec_sft.preflight import (
    _internal_cutoff,
    _exceeds_internal_cutoff,
    _record_count_blocker,
    _validate_locked_paths,
)


class PreflightLockBindingTest(unittest.TestCase):
    def test_model_and_data_paths_must_equal_the_verified_lock(self) -> None:
        integrity = {
            "model_root": "/locked/model",
            "derived_dataset": {"path": "/locked/data.jsonl", "records": 63_700},
        }
        _validate_locked_paths(
            Path("/locked/model"), Path("/locked/data.jsonl"), integrity
        )
        with self.assertRaisesRegex(RuntimeError, "model path"):
            _validate_locked_paths(
                Path("/other/model"), Path("/locked/data.jsonl"), integrity
            )
        with self.assertRaisesRegex(RuntimeError, "data path"):
            _validate_locked_paths(
                Path("/locked/model"), Path("/other/data.jsonl"), integrity
            )

    def test_actual_record_count_must_equal_the_lock(self) -> None:
        integrity = {"derived_dataset": {"records": 63_700}}
        self.assertIsNone(_record_count_blocker(integrity, 63_700))
        blocker = _record_count_blocker(integrity, 63_699)
        self.assertIsNotNone(blocker)
        self.assertIn("63700", blocker or "")
        self.assertIn("63699", blocker or "")

    def test_requested_cutoff_uses_llama_factory_internal_cutoff(self) -> None:
        self.assertEqual(_internal_cutoff(16_384), 16_383)
        self.assertFalse(_exceeds_internal_cutoff(16_383, 16_384))
        self.assertTrue(_exceeds_internal_cutoff(16_384, 16_384))
        self.assertEqual(_internal_cutoff(32_768), 32_767)
        with self.assertRaises(ValueError):
            _internal_cutoff(1)


if __name__ == "__main__":
    unittest.main()
