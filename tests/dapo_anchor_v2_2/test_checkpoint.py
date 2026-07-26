from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ksllm4rec_dapo_anchor_v2_2._infra.dapo_checkpoint import (
    RecoveryCursor,
    load_latest_recovery,
    recovery_checkpoint_due,
    restore_training_state,
    save_recovery_checkpoint,
)
from ksllm4rec_dapo_anchor_v2_2._infra.dapo_sampling import SourceCursor
from ksllm4rec_dapo_anchor_v2_2._infra.rloo_integrity import canonical_sha256
from ksllm4rec_dapo_anchor_v2_2.config import build_config
from ksllm4rec_dapo_anchor_v2_2.trainer import WindowLRScheduler


class _Bundle:
    def save_policy(self, output_dir: Path):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        (output / "adapter_model.safetensors").write_bytes(b"policy")
        (output / "adapter_config.json").write_text(
            '{"r":64,"lora_alpha":64}', encoding="utf-8"
        )


def _signature() -> dict:
    inputs = {"run": "dapo-anchor-v2.2"}
    return {"sha256": canonical_sha256(inputs), "inputs": inputs}


def _cursor(window: int = 11, anchor_steps: int = 1) -> RecoveryCursor:
    return RecoveryCursor(
        next_window_index=window,
        rl_optimizer_update_step=window * 4,
        anchor_optimizer_update_step=anchor_steps,
        optimizer_update_step=window * 4 + anchor_steps,
        source_cursor=SourceCursor(cycle_index=0, offset=123),
        window_log_rows=window,
        group_log_rows=window * 32,
    )


class RecoveryCursorV22Test(unittest.TestCase):
    def test_first_completed_window_may_have_one_anchor_step(self) -> None:
        cursor = _cursor(window=1, anchor_steps=1)
        self.assertEqual(cursor.anchor_optimizer_update_step, 1)
        self.assertEqual(cursor.optimizer_update_step, 5)

    def test_cursor_tracks_rl_and_anchor_steps_separately(self) -> None:
        cursor = RecoveryCursor(
            next_window_index=11,
            rl_optimizer_update_step=44,
            anchor_optimizer_update_step=1,
            optimizer_update_step=45,
            source_cursor=SourceCursor(cycle_index=0, offset=123),
            window_log_rows=11,
            group_log_rows=11 * 32,
        )
        self.assertEqual(cursor.rl_optimizer_update_step, 44)
        self.assertEqual(cursor.anchor_optimizer_update_step, 1)
        self.assertEqual(cursor.optimizer_update_step, 45)

    def test_cursor_rejects_inconsistent_total_step(self) -> None:
        with self.assertRaisesRegex(ValueError, "sum"):
            RecoveryCursor(
                next_window_index=11,
                rl_optimizer_update_step=44,
                anchor_optimizer_update_step=1,
                optimizer_update_step=44,
                source_cursor=SourceCursor(cycle_index=0, offset=123),
                window_log_rows=11,
                group_log_rows=11 * 32,
            )

    def test_checkpoint_round_trip_restores_scheduler_and_rng(self) -> None:
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=0.0)
        scheduler = WindowLRScheduler(
            optimizer, build_config(), completed_windows=11
        )
        parameter.square().backward()
        optimizer.step()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_recovery_checkpoint(
                _Bundle(), optimizer, scheduler, _cursor(), root, _signature()
            )
            expected = (random.random(), float(np.random.rand()), torch.rand(3))
            loaded = load_latest_recovery(root, _signature())
            self.assertIsNotNone(loaded)
            _, cursor, state = loaded
            self.assertEqual(cursor, _cursor())

            restored_parameter = torch.nn.Parameter(torch.tensor(1.0))
            restored_optimizer = torch.optim.AdamW([restored_parameter], lr=0.0)
            restored_scheduler = WindowLRScheduler(
                restored_optimizer, build_config()
            )
            restore_training_state(restored_optimizer, restored_scheduler, state)
            actual = (random.random(), float(np.random.rand()), torch.rand(3))
            self.assertEqual(actual[0], expected[0])
            self.assertEqual(actual[1], expected[1])
            torch.testing.assert_close(actual[2], expected[2])
            self.assertEqual(restored_scheduler.completed_windows, 11)

    def test_checkpoint_due_is_counted_in_complete_windows(self) -> None:
        self.assertTrue(
            recovery_checkpoint_due(
                _cursor(window=25, anchor_steps=15),
                epoch_finished=False,
                interval_windows=25,
            )
        )
        self.assertFalse(
            recovery_checkpoint_due(
                _cursor(window=24, anchor_steps=14),
                epoch_finished=False,
                interval_windows=25,
            )
        )

    def test_cursor_rejects_more_than_one_anchor_step_per_completed_window(self) -> None:
        with self.assertRaisesRegex(ValueError, "one anchor"):
            RecoveryCursor(
                next_window_index=11,
                rl_optimizer_update_step=44,
                anchor_optimizer_update_step=12,
                optimizer_update_step=56,
                source_cursor=SourceCursor(cycle_index=0, offset=123),
                window_log_rows=11,
                group_log_rows=11 * 32,
            )


if __name__ == "__main__":
    unittest.main()
