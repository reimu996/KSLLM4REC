from __future__ import annotations

import json
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ksllm4rec_rloo.checkpoint import (
    RecoveryCursor,
    capture_rng_state,
    load_latest_recovery,
    recovery_checkpoint_due,
    restore_rng_state,
    restore_training_state,
    save_recovery_checkpoint,
)
from ksllm4rec_rloo.integrity import canonical_sha256


class ToyBundle:
    def save_policy(self, output_dir: Path) -> tuple[Path, Path]:
        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        weights = output / "adapter_model.safetensors"
        config = output / "adapter_config.json"
        weights.write_bytes(b"r64-policy")
        config.write_text(
            '{"r":64,"lora_alpha":64,"lora_dropout":0.0}',
            encoding="utf-8",
        )
        (output / "nested").mkdir()
        (output / "nested" / "policy-metadata.json").write_text(
            '{"format":"toy"}', encoding="utf-8"
        )
        return weights, config


def signature(label: str = "run-a") -> dict:
    inputs = {"run": label, "runtime_code": {"sha256": "abc"}}
    return {"sha256": canonical_sha256(inputs), "inputs": inputs}


def cursor(
    *,
    window_step: int = 100,
    optimizer_update_step: int = 100,
    next_group_offset: int = 800,
    groups_completed: int = 800,
) -> RecoveryCursor:
    return RecoveryCursor(
        epoch_index=0,
        next_group_offset=next_group_offset,
        window_step=window_step,
        optimizer_update_step=optimizer_update_step,
        groups_completed=groups_completed,
        rollout_chunk=8,
        loss_chunk=8,
        lambda0=0.02,
    )


RESOLVED_CONTRACT = {
    "spec_version": "2.0",
    "anchor": {"lambda0": 0.02},
    "reference": None,
}


class RecoveryCursorTest(unittest.TestCase):
    def test_cursor_contains_independent_window_and_optimizer_steps(self) -> None:
        value = cursor(window_step=12, optimizer_update_step=9)
        self.assertEqual(value.window_step, 12)
        self.assertEqual(value.optimizer_update_step, 9)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            cursor(window_step=9, optimizer_update_step=12)

    def test_checkpoint_schedule_is_update_and_epoch_driven(self) -> None:
        self.assertTrue(
            recovery_checkpoint_due(
                cursor(), optimizer_stepped=True, epoch_finished=False
            )
        )
        self.assertFalse(
            recovery_checkpoint_due(
                cursor(optimizer_update_step=99),
                optimizer_stepped=True,
                epoch_finished=False,
            )
        )
        self.assertFalse(
            recovery_checkpoint_due(
                cursor(), optimizer_stepped=False, epoch_finished=False
            )
        )
        self.assertTrue(
            recovery_checkpoint_due(
                cursor(optimizer_update_step=99),
                optimizer_stepped=False,
                epoch_finished=True,
            )
        )


class RngRecoveryTest(unittest.TestCase):
    def test_all_rng_streams_restore_exactly(self) -> None:
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

    def test_round_trip_restores_cursor_optimizer_scheduler_and_next_rng(self) -> None:
        random.seed(17)
        np.random.seed(17)
        torch.manual_seed(17)
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda step: 1.0 - min(step, 10) / 20
        )
        parameter.square().sum().backward()
        optimizer.step()
        scheduler.step()

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = save_recovery_checkpoint(
                ToyBundle(),
                optimizer,
                scheduler,
                cursor(),
                root,
                signature(),
                RESOLVED_CONTRACT,
            )
            expected_rng = (random.random(), float(np.random.rand()), torch.rand(4))
            random.random()
            np.random.rand()
            torch.rand(4)

            loaded = load_latest_recovery(root, signature(), RESOLVED_CONTRACT)
            self.assertIsNotNone(loaded)
            loaded_path, loaded_cursor, training = loaded
            self.assertEqual(loaded_path, checkpoint)
            self.assertEqual(loaded_cursor, cursor())
            self.assertEqual(training["resolved_contract"], RESOLVED_CONTRACT)

            new_parameter = torch.nn.Parameter(torch.tensor([1.0]))
            new_optimizer = torch.optim.AdamW([new_parameter], lr=1e-3)
            new_scheduler = torch.optim.lr_scheduler.LambdaLR(
                new_optimizer, lambda step: 1.0 - min(step, 10) / 20
            )
            restore_training_state(new_optimizer, new_scheduler, training)
            self.assertEqual(
                new_scheduler.state_dict()["last_epoch"],
                scheduler.state_dict()["last_epoch"],
            )
            actual_rng = (random.random(), float(np.random.rand()), torch.rand(4))
            self.assertEqual(actual_rng[0], expected_rng[0])
            self.assertEqual(actual_rng[1], expected_rng[1])
            torch.testing.assert_close(actual_rng[2], expected_rng[2])

    def test_optional_scheduler_round_trips_as_none(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            save_recovery_checkpoint(
                ToyBundle(),
                optimizer,
                None,
                cursor(),
                root,
                signature(),
                RESOLVED_CONTRACT,
            )
            _, _, training = load_latest_recovery(
                root, signature(), RESOLVED_CONTRACT
            )
            restored_parameter = torch.nn.Parameter(torch.tensor([1.0]))
            restored_optimizer = torch.optim.AdamW([restored_parameter], lr=1e-3)
            restore_training_state(restored_optimizer, None, training)
            self.assertIsNone(training["scheduler"])


class CheckpointIntegrityTest(unittest.TestCase):
    def make_checkpoint(self, root: Path, value: RecoveryCursor | None = None) -> Path:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        return save_recovery_checkpoint(
            ToyBundle(),
            optimizer,
            None,
            value or cursor(),
            root,
            signature(),
            RESOLVED_CONTRACT,
        )

    def test_every_payload_file_is_hashed_and_tampering_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.make_checkpoint(root)
            manifest = json.loads(
                (checkpoint / "manifest.json").read_text(encoding="utf-8")
            )
            self.assertIn("nested/policy-metadata.json", manifest["files"])
            (checkpoint / "nested" / "policy-metadata.json").write_text(
                "tampered", encoding="utf-8"
            )
            with self.assertRaisesRegex(RuntimeError, "SHA256/size mismatch"):
                load_latest_recovery(root, signature(), RESOLVED_CONTRACT)

    def test_extra_file_is_rejected_instead_of_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint = self.make_checkpoint(root)
            (checkpoint / "untracked.bin").write_bytes(b"extra")
            with self.assertRaisesRegex(RuntimeError, "file set mismatch"):
                load_latest_recovery(root, signature(), RESOLVED_CONTRACT)

    def test_contract_signature_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_checkpoint(root)
            with self.assertRaisesRegex(RuntimeError, "different runtime contract"):
                load_latest_recovery(
                    root, signature("run-b"), RESOLVED_CONTRACT
                )

    def test_expected_resolved_contract_mismatch_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_checkpoint(root)
            with self.assertRaisesRegex(RuntimeError, "different resolved contract"):
                load_latest_recovery(
                    root,
                    signature(),
                    {**RESOLVED_CONTRACT, "anchor": {"lambda0": 0.03}},
                )

    def test_cursor_lambda0_must_equal_frozen_contract(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor([1.0]))
        optimizer = torch.optim.AdamW([parameter], lr=1e-3)
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "lambda0"):
                save_recovery_checkpoint(
                    ToyBundle(),
                    optimizer,
                    None,
                    cursor(),
                    Path(directory),
                    signature(),
                    {"lambda0": 0.03},
                )

    def test_complete_orphan_is_promoted_after_pointer_crash(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_checkpoint(root, cursor())
            first_pointer = (root / "latest.json").read_text(encoding="utf-8")
            second = cursor(
                window_step=200,
                optimizer_update_step=199,
                next_group_offset=1600,
                groups_completed=1600,
            )
            second_path = self.make_checkpoint(root, second)
            (root / "latest.json").write_text(first_pointer, encoding="utf-8")

            checkpoint, loaded_cursor, _ = load_latest_recovery(
                root, signature(), RESOLVED_CONTRACT
            )

            self.assertEqual(checkpoint, second_path)
            self.assertEqual(loaded_cursor, second)
            pointer = json.loads((root / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(pointer["window_step"], 200)

    def test_corrupt_newest_checkpoint_never_falls_back(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.make_checkpoint(root, cursor())
            newest = self.make_checkpoint(
                root,
                cursor(
                    window_step=200,
                    optimizer_update_step=199,
                    next_group_offset=1600,
                    groups_completed=1600,
                ),
            )
            (newest / "cursor.json").write_text("corrupt", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "SHA256/size mismatch"):
                load_latest_recovery(root, signature(), RESOLVED_CONTRACT)


if __name__ == "__main__":
    unittest.main()
