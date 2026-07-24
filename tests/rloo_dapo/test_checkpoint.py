from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ksllm4rec_rloo.integrity import canonical_sha256
from ksllm4rec_rloo_dapo.checkpoint import (
    RecoveryCursor,
    load_latest_recovery,
    recovery_checkpoint_due,
    restore_training_state,
    save_recovery_checkpoint,
    validate_recovery_checkpoint,
)
from ksllm4rec_rloo_dapo.config import approved_config
from ksllm4rec_rloo_dapo.sampling import SourceCursor
from ksllm4rec_rloo_dapo.trainer import WindowLRScheduler


class _Bundle:
    def save_policy(self, output_dir: Path):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        weights = output / "adapter_model.safetensors"
        config = output / "adapter_config.json"
        weights.write_bytes(b"policy")
        config.write_text('{"r":64,"lora_alpha":64}', encoding="utf-8")
        return weights, config


def _signature(label: str = "v2") -> dict:
    inputs = {"run": label}
    return {"sha256": canonical_sha256(inputs), "inputs": inputs}


def _cursor(window: int = 25) -> RecoveryCursor:
    return RecoveryCursor(
        next_window_index=window,
        optimizer_update_step=window * 4,
        source_cursor=SourceCursor(cycle_index=2, offset=17),
        window_log_rows=window,
        group_log_rows=window * 32,
    )


class RecoveryTest(unittest.TestCase):
    def test_checkpoint_is_due_only_on_complete_window_boundaries(self) -> None:
        self.assertTrue(recovery_checkpoint_due(_cursor(25), epoch_finished=False))
        self.assertFalse(recovery_checkpoint_due(_cursor(24), epoch_finished=False))
        self.assertTrue(recovery_checkpoint_due(_cursor(24), epoch_finished=True))

    def test_round_trip_restores_all_training_state(self) -> None:
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.0)
        scheduler = WindowLRScheduler(
            optimizer, approved_config(), completed_windows=25
        )
        parameter.square().backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = save_recovery_checkpoint(
                _Bundle(), optimizer, scheduler, _cursor(), root, _signature()
            )
            expected = (random.random(), float(np.random.rand()), torch.rand(3))
            loaded = load_latest_recovery(root, _signature())
            self.assertIsNotNone(loaded)
            _, cursor, state = loaded
            self.assertEqual(cursor, _cursor())
            validated_cursor, _ = validate_recovery_checkpoint(checkpoint, _signature())
            self.assertEqual(validated_cursor, _cursor())

            restored_parameter = torch.nn.Parameter(torch.tensor(1.0))
            restored_optimizer = torch.optim.AdamW([restored_parameter], lr=0.0)
            restored_scheduler = WindowLRScheduler(
                restored_optimizer, approved_config()
            )
            restore_training_state(restored_optimizer, restored_scheduler, state)
            actual = (random.random(), float(np.random.rand()), torch.rand(3))
            self.assertEqual(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])
            torch.testing.assert_close(actual[2], expected[2])
            self.assertEqual(restored_scheduler.completed_windows, 25)

    def test_tampering_is_rejected(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.0)
        scheduler = WindowLRScheduler(optimizer, approved_config())
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = save_recovery_checkpoint(
                _Bundle(), optimizer, scheduler, _cursor(0), root, _signature()
            )
            (checkpoint / "adapter_model.safetensors").write_bytes(b"tampered")
            with self.assertRaisesRegex(RuntimeError, "SHA256/size mismatch"):
                load_latest_recovery(root, _signature())


if __name__ == "__main__":
    unittest.main()
