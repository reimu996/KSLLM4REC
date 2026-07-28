from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ksllm4rec_dapo_anchor_multitask_v1._infra.rloo_integrity import (
    canonical_sha256,
)
from ksllm4rec_dapo_anchor_multitask_v1.checkpoint import (
    PendingBuffer,
    RecoveryState,
    load_latest_recovery,
    restore_training_state,
    save_recovery_checkpoint,
)


class _Bundle:
    def save_policy(self, output_dir: Path) -> tuple[Path, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        weights = output_dir / "adapter_model.safetensors"
        config = output_dir / "adapter_config.json"
        weights.write_bytes(b"multitask-policy")
        config.write_text('{"r":64,"lora_alpha":64}\n', encoding="utf-8")
        return weights, config


class _FailingBundle:
    def save_policy(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "adapter_model.safetensors").write_bytes(b"partial")
        raise RuntimeError("simulated policy save interruption")


def _signature() -> dict[str, object]:
    inputs = {"spec": "DAPO-Anchor-Multitask-V1.0", "arm": "C"}
    return {"sha256": canonical_sha256(inputs), "inputs": inputs}


def _optimizer_and_scheduler() -> tuple[torch.optim.Optimizer, object]:
    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    optimizer = torch.optim.AdamW([parameter], lr=1.0e-6)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1.0)
    parameter.grad = torch.tensor([0.25])
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad(set_to_none=True)
    return optimizer, scheduler


class CheckpointTests(unittest.TestCase):
    def test_atomic_checkpoint_roundtrip_restores_full_interrupted_state(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            optimizer, scheduler = _optimizer_and_scheduler()
            pending = PendingBuffer(
                snapshot_policy_step=7,
                payload={
                    "source_blocks": [13, 14],
                    "recommendation_group_ids": ["rec-13"],
                    "text_group_ids": ["text-14"],
                    "old_log_probs": torch.tensor([-0.2, -0.4]),
                },
            )
            recovery = RecoveryState(
                next_source_block=15,
                optimization_window_index=3,
                completed_policy_steps=7,
                total_optimizer_steps=7,
                final_auxiliary_flush_steps=0,
                pending_buffer=pending,
                last_rl_reference_grad_norm=0.75,
                source_log_rows=15,
                group_log_rows=781,
                window_log_rows=3,
            )

            random.seed(101)
            np.random.seed(202)
            torch.manual_seed(303)
            checkpoint = save_recovery_checkpoint(
                _Bundle(),
                optimizer,
                scheduler,
                recovery,
                root / "recovery",
                _signature(),
            )
            expected_rng = (
                random.random(),
                float(np.random.rand()),
                float(torch.rand(())),
            )

            random.seed(1)
            np.random.seed(2)
            torch.manual_seed(3)
            restored_optimizer, restored_scheduler = _optimizer_and_scheduler()
            loaded = load_latest_recovery(root / "recovery", _signature())

            self.assertIsNotNone(loaded)
            assert loaded is not None
            loaded_path, loaded_recovery, training_state = loaded
            self.assertEqual(loaded_path, checkpoint)
            self.assertEqual(loaded_recovery.next_source_block, 15)
            self.assertEqual(loaded_recovery.optimization_window_index, 3)
            self.assertEqual(loaded_recovery.completed_policy_steps, 7)
            self.assertEqual(loaded_recovery.total_optimizer_steps, 7)
            self.assertEqual(loaded_recovery.final_auxiliary_flush_steps, 0)
            self.assertIsNone(loaded_recovery.final_auxiliary_flush)
            self.assertEqual(loaded_recovery.last_rl_reference_grad_norm, 0.75)
            self.assertEqual(loaded_recovery.source_log_rows, 15)
            self.assertEqual(loaded_recovery.group_log_rows, 781)
            self.assertEqual(loaded_recovery.window_log_rows, 3)
            self.assertIsNotNone(loaded_recovery.pending_buffer)
            assert loaded_recovery.pending_buffer is not None
            self.assertEqual(loaded_recovery.pending_buffer.snapshot_policy_step, 7)
            self.assertEqual(
                loaded_recovery.pending_buffer.payload["source_blocks"], [13, 14]
            )
            self.assertTrue(
                torch.equal(
                    loaded_recovery.pending_buffer.payload["old_log_probs"],
                    torch.tensor([-0.2, -0.4]),
                )
            )

            restore_training_state(
                restored_optimizer, restored_scheduler, training_state
            )
            actual_rng = (
                random.random(),
                float(np.random.rand()),
                float(torch.rand(())),
            )
            self.assertEqual(actual_rng, expected_rng)
            self.assertEqual(restored_optimizer.state_dict(), optimizer.state_dict())
            self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())

    def test_final_auxiliary_audit_roundtrips_with_its_atomic_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            optimizer, scheduler = _optimizer_and_scheduler()
            audit = {
                "optimizer_steps": 1,
                "anchor_groups": 7,
                "lambda_effective": 0.01,
            }
            recovery = RecoveryState(
                next_source_block=532,
                optimization_window_index=9,
                completed_policy_steps=36,
                total_optimizer_steps=37,
                final_auxiliary_flush_steps=1,
                pending_buffer=None,
                last_rl_reference_grad_norm=0.75,
                source_log_rows=27_613,
                group_log_rows=251,
                window_log_rows=9,
                final_auxiliary_flush=audit,
            )

            save_recovery_checkpoint(
                _Bundle(),
                optimizer,
                scheduler,
                recovery,
                Path(directory) / "recovery",
                _signature(),
            )
            loaded = load_latest_recovery(
                Path(directory) / "recovery", _signature()
            )

            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded[1].final_auxiliary_flush, audit)

    def test_corrupt_newest_checkpoint_is_rejected_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "recovery"
            optimizer, scheduler = _optimizer_and_scheduler()
            older = RecoveryState(
                next_source_block=8,
                optimization_window_index=1,
                completed_policy_steps=4,
                total_optimizer_steps=4,
                final_auxiliary_flush_steps=0,
                pending_buffer=None,
                last_rl_reference_grad_norm=0.5,
                source_log_rows=8,
                group_log_rows=224,
                window_log_rows=1,
            )
            newest = RecoveryState(
                next_source_block=16,
                optimization_window_index=2,
                completed_policy_steps=8,
                total_optimizer_steps=8,
                final_auxiliary_flush_steps=0,
                pending_buffer=None,
                last_rl_reference_grad_norm=0.6,
                source_log_rows=16,
                group_log_rows=448,
                window_log_rows=2,
            )
            save_recovery_checkpoint(
                _Bundle(), optimizer, scheduler, older, root, _signature()
            )
            newest_path = save_recovery_checkpoint(
                _Bundle(), optimizer, scheduler, newest, root, _signature()
            )
            (newest_path / "adapter_model.safetensors").write_bytes(b"tampered")

            with self.assertRaisesRegex(RuntimeError, "mismatch"):
                load_latest_recovery(root, _signature())

    def test_interrupted_pending_buffer_must_match_snapshot_policy_step(self) -> None:
        with self.assertRaisesRegex(ValueError, "snapshot_policy_step"):
            RecoveryState(
                next_source_block=15,
                optimization_window_index=3,
                completed_policy_steps=7,
                total_optimizer_steps=7,
                final_auxiliary_flush_steps=0,
                pending_buffer=PendingBuffer(
                    snapshot_policy_step=6,
                    payload={"source_blocks": [13, 14]},
                ),
                last_rl_reference_grad_norm=0.75,
                source_log_rows=15,
                group_log_rows=781,
                window_log_rows=3,
            )

    def test_interrupted_save_never_publishes_a_partial_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "recovery"
            optimizer, scheduler = _optimizer_and_scheduler()
            complete = RecoveryState(
                next_source_block=8,
                optimization_window_index=1,
                completed_policy_steps=4,
                total_optimizer_steps=4,
                final_auxiliary_flush_steps=0,
                pending_buffer=None,
                last_rl_reference_grad_norm=0.5,
                source_log_rows=8,
                group_log_rows=224,
                window_log_rows=1,
            )
            save_recovery_checkpoint(
                _Bundle(), optimizer, scheduler, complete, root, _signature()
            )
            interrupted = RecoveryState(
                next_source_block=16,
                optimization_window_index=2,
                completed_policy_steps=8,
                total_optimizer_steps=8,
                final_auxiliary_flush_steps=0,
                pending_buffer=None,
                last_rl_reference_grad_norm=0.6,
                source_log_rows=16,
                group_log_rows=448,
                window_log_rows=2,
            )

            with self.assertRaisesRegex(RuntimeError, "simulated"):
                save_recovery_checkpoint(
                    _FailingBundle(),
                    optimizer,
                    scheduler,
                    interrupted,
                    root,
                    _signature(),
                )

            self.assertFalse(
                any(path.name.startswith(".checkpoint-") for path in root.iterdir())
            )
            loaded = load_latest_recovery(root, _signature())
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded[1].next_source_block, 8)


if __name__ == "__main__":
    unittest.main()
