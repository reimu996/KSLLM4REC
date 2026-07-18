import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ksllm4rec_grpo.gates import load_training_gates


class GateBindingTest(unittest.TestCase):
    def setUp(self):
        self.config = {
            "memory": {"chunk_candidates": [8, 4, 2, 1], "max_reserved_gib": 20.0},
            "gates": {
                "signal_groups": 512,
                "timing_groups": 32,
                "max_projected_hours": 48.0,
            },
        }
        self.signature = {"sha256": "runtime", "inputs": {}}

    def _write_reports(self, root: Path, signal_chunk: int = 8) -> tuple[Path, ...]:
        memory = {
            "runtime_signature": self.signature,
            "rollout_chunk": 8,
            "loss_chunk": 8,
            "attempts": [
                {
                    "success": True,
                    "phase": "loss",
                    "rollout_chunk": 8,
                    "chunk": 8,
                    "peak_reserved_gib": 6.0,
                }
            ],
        }
        signal = {
            "runtime_signature": self.signature,
            "passed": True,
            "groups_run": 512,
            "gradient_norm_max": 2.0,
            "parameter_max_change_this_invocation": 0.1,
            "state": {
                "rollout_chunk": signal_chunk,
                "loss_chunk": signal_chunk,
            },
        }
        timing = {
            "runtime_signature": self.signature,
            "passed": True,
            "groups_run": 32,
            "projected_two_epoch_hours": 30.0,
            "state": {"rollout_chunk": 8, "loss_chunk": 8},
        }
        paths = tuple(
            root / name for name in ("memory.json", "signal.json", "timing.json")
        )
        for path, value in zip(paths, (memory, signal, timing), strict=True):
            path.write_text(json.dumps(value), encoding="utf-8")
        return paths

    def test_accepts_three_reports_for_one_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            memory, signal, timing = self._write_reports(Path(directory))
            with patch(
                "ksllm4rec_grpo.gates.runtime_signature",
                return_value=self.signature,
            ):
                chunks = load_training_gates(
                    config=self.config,
                    groups_path=Path("groups"),
                    trie_dir=Path("trie"),
                    memory_path=memory,
                    signal_path=signal,
                    timing_path=timing,
                )
        self.assertEqual(chunks, (8, 8))

    def test_rejects_signal_report_for_another_chunk(self):
        with tempfile.TemporaryDirectory() as directory:
            memory, signal, timing = self._write_reports(
                Path(directory), signal_chunk=4
            )
            with patch(
                "ksllm4rec_grpo.gates.runtime_signature",
                return_value=self.signature,
            ):
                with self.assertRaisesRegex(RuntimeError, "Signal gate"):
                    load_training_gates(
                        config=self.config,
                        groups_path=Path("groups"),
                        trie_dir=Path("trie"),
                        memory_path=memory,
                        signal_path=signal,
                        timing_path=timing,
                    )


if __name__ == "__main__":
    unittest.main()
