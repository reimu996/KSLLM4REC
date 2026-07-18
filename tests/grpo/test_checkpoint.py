import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ksllm4rec_grpo.checkpoint import (
    RecoveryCursor,
    capture_rng_state,
    load_latest_recovery,
    restore_rng_state,
    save_recovery_checkpoint,
)


class ToyBundle:
    def save_policy(self, output_dir):
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        weights = output / "adapter_model.safetensors"
        config = output / "adapter_config.json"
        weights.write_bytes(b"weights")
        config.write_text('{"lora_dropout":0.0}', encoding="utf-8")
        return weights, config


CONTRACT = {"sha256": "toy", "inputs": {}}


class RngRecoveryTest(unittest.TestCase):
    def test_all_rng_streams_restore_exactly(self):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        state = capture_rng_state()
        expected = (random.random(), float(np.random.rand()), torch.rand(3))
        random.random()
        np.random.rand()
        torch.rand(3)
        restore_rng_state(state)
        actual = (random.random(), float(np.random.rand()), torch.rand(3))
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2])

    def test_atomic_checkpoint_round_trips_cursor_optimizer_and_scheduler(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        (parameter.square().sum()).backward()
        optimizer.step()
        scheduler.step()
        cursor = RecoveryCursor(1, 800, 100, 800, 8, 8)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = save_recovery_checkpoint(
                ToyBundle(), optimizer, scheduler, cursor, root, CONTRACT
            )
            loaded = load_latest_recovery(root, CONTRACT)
            self.assertIsNotNone(loaded)
            loaded_path, loaded_cursor, training = loaded
            self.assertEqual(loaded_path, checkpoint)
            self.assertEqual(loaded_cursor, cursor)
            self.assertEqual(
                training["scheduler"]["last_epoch"],
                scheduler.state_dict()["last_epoch"],
            )

            (checkpoint / "cursor.json").write_text("corrupt", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "corrupt"):
                load_latest_recovery(root, CONTRACT)

    def test_complete_orphan_checkpoint_is_promoted_after_pointer_crash(self):
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = RecoveryCursor(1, 800, 100, 800, 8, 8)
            save_recovery_checkpoint(
                ToyBundle(), optimizer, scheduler, first, root, CONTRACT
            )
            second = RecoveryCursor(1, 1600, 200, 1600, 8, 8)
            save_recovery_checkpoint(
                ToyBundle(), optimizer, scheduler, second, root, CONTRACT
            )
            (root / "latest.json").write_text(
                '{"checkpoint":"checkpoint-step-000100","global_step":100}',
                encoding="utf-8",
            )

            checkpoint, cursor, _ = load_latest_recovery(root, CONTRACT)

            self.assertEqual(checkpoint.name, "checkpoint-step-000200")
            self.assertEqual(cursor, second)
            latest = json.loads((root / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(latest["global_step"], 200)


if __name__ == "__main__":
    unittest.main()
