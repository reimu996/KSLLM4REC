from __future__ import annotations

import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from ksllm4rec_dapo_anchor_multitask_v1_1._infra.rloo_integrity import (
    canonical_sha256,
)
from ksllm4rec_dapo_anchor_multitask_v1_1.checkpoint import (
    PendingBuffer,
    RecoveryState,
    load_latest_recovery,
    restore_training_state,
    save_recovery_checkpoint,
)
from ksllm4rec_dapo_anchor_multitask_v1_1.trainer import _truncate_jsonl


_PLAN_SHA = "a" * 64
_EPOCH_PLAN_SHAS = ("b" * 64, "c" * 64)


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
    inputs = {"spec": "DAPO-Anchor-Multitask-V1.1-E2", "arm": "C"}
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


def _recovery(
    *,
    next_source_block: int,
    optimization_window_index: int,
    completed_policy_steps: int,
    auxiliary_flush_steps: int = 0,
    pending_buffer: PendingBuffer | None = None,
    last_rl_reference_grad_norm: float | None = 0.75,
    source_log_rows: int | None = None,
    group_log_rows: int = 781,
    window_log_rows: int | None = None,
    anchor_log_rows: int = 781,
    epoch_auxiliary_flushes: tuple[dict[str, object], ...] = (),
) -> RecoveryState:
    epoch_index, next_block_in_epoch = divmod(next_source_block, 532)
    return RecoveryState(
        next_source_block=next_source_block,
        epoch_index=epoch_index,
        next_block_in_epoch=next_block_in_epoch,
        source_plan_sha256=_PLAN_SHA,
        epoch_plan_sha256s=_EPOCH_PLAN_SHAS,
        optimization_window_index=optimization_window_index,
        completed_policy_steps=completed_policy_steps,
        total_optimizer_steps=completed_policy_steps + auxiliary_flush_steps,
        auxiliary_flush_steps=auxiliary_flush_steps,
        pending_buffer=pending_buffer,
        last_rl_reference_grad_norm=last_rl_reference_grad_norm,
        source_log_rows=(next_source_block if source_log_rows is None else source_log_rows),
        group_log_rows=group_log_rows,
        window_log_rows=(
            optimization_window_index if window_log_rows is None else window_log_rows
        ),
        anchor_log_rows=anchor_log_rows,
        epoch_auxiliary_flushes=epoch_auxiliary_flushes,
    )


class CheckpointTests(unittest.TestCase):
    def test_zero_row_jsonl_is_materialized_for_the_recovery_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "epoch_auxiliary_flushes.jsonl"
            _truncate_jsonl(path, 0)
            self.assertTrue(path.is_file())
            self.assertEqual(path.read_bytes(), b"")
            _truncate_jsonl(path, 0)
            self.assertEqual(path.read_bytes(), b"")

    def test_atomic_checkpoint_roundtrip_restores_full_epoch_aware_state(self) -> None:
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
            recovery = _recovery(
                next_source_block=15,
                optimization_window_index=3,
                completed_policy_steps=7,
                pending_buffer=pending,
                source_log_rows=15,
                group_log_rows=781,
                window_log_rows=3,
                anchor_log_rows=15,
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
            self.assertEqual(loaded_recovery.epoch_index, 0)
            self.assertEqual(loaded_recovery.next_block_in_epoch, 15)
            self.assertEqual(loaded_recovery.source_plan_sha256, _PLAN_SHA)
            self.assertEqual(loaded_recovery.epoch_plan_sha256s, _EPOCH_PLAN_SHAS)
            self.assertEqual(loaded_recovery.optimization_window_index, 3)
            self.assertEqual(loaded_recovery.completed_policy_steps, 7)
            self.assertEqual(loaded_recovery.total_optimizer_steps, 7)
            self.assertEqual(loaded_recovery.auxiliary_flush_steps, 0)
            self.assertEqual(loaded_recovery.epoch_auxiliary_flushes, ())
            self.assertEqual(loaded_recovery.last_rl_reference_grad_norm, 0.75)
            self.assertEqual(loaded_recovery.source_log_rows, 15)
            self.assertEqual(loaded_recovery.group_log_rows, 781)
            self.assertEqual(loaded_recovery.window_log_rows, 3)
            self.assertEqual(loaded_recovery.anchor_log_rows, 15)
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

    def test_epoch_boundary_flush_audit_roundtrips_at_epoch_one_cursor(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            optimizer, scheduler = _optimizer_and_scheduler()
            audit = {
                "epoch_index": 0,
                "optimizer_steps": 1,
                "anchor_groups": 7,
                "lambda_effective": 0.01,
                "scheduler_advanced": False,
            }
            recovery = _recovery(
                next_source_block=532,
                optimization_window_index=9,
                completed_policy_steps=36,
                auxiliary_flush_steps=1,
                source_log_rows=27_613,
                group_log_rows=251,
                window_log_rows=9,
                anchor_log_rows=27_613,
                epoch_auxiliary_flushes=(audit,),
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
            self.assertEqual(loaded[1].epoch_index, 1)
            self.assertEqual(loaded[1].next_block_in_epoch, 0)
            self.assertEqual(loaded[1].epoch_auxiliary_flushes, (audit,))
            self.assertEqual(loaded[1].auxiliary_flush_steps, 1)

    def test_cursor_rejects_old_one_epoch_boundary_encoding(self) -> None:
        with self.assertRaisesRegex(ValueError, "Epoch boundary checkpoints"):
            RecoveryState(
                next_source_block=532,
                epoch_index=0,
                next_block_in_epoch=532,
                source_plan_sha256=_PLAN_SHA,
                epoch_plan_sha256s=_EPOCH_PLAN_SHAS,
                optimization_window_index=9,
                completed_policy_steps=36,
                total_optimizer_steps=36,
                auxiliary_flush_steps=0,
                pending_buffer=None,
                last_rl_reference_grad_norm=0.75,
                source_log_rows=27_613,
                group_log_rows=251,
                window_log_rows=9,
                anchor_log_rows=27_613,
            )

    def test_completed_two_epoch_cursor_is_1064_epoch_two_block_zero(self) -> None:
        state = _recovery(
            next_source_block=1_064,
            optimization_window_index=400,
            completed_policy_steps=1_600,
            source_log_rows=55_226,
            group_log_rows=20_000,
            window_log_rows=400,
            anchor_log_rows=55_226,
        )
        self.assertEqual(state.epoch_index, 2)
        self.assertEqual(state.next_block_in_epoch, 0)

    def test_corrupt_newest_checkpoint_is_rejected_without_fallback(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "recovery"
            optimizer, scheduler = _optimizer_and_scheduler()
            older = _recovery(
                next_source_block=8,
                optimization_window_index=1,
                completed_policy_steps=4,
            )
            newest = _recovery(
                next_source_block=16,
                optimization_window_index=2,
                completed_policy_steps=8,
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
            _recovery(
                next_source_block=15,
                optimization_window_index=3,
                completed_policy_steps=7,
                pending_buffer=PendingBuffer(
                    snapshot_policy_step=6,
                    payload={"source_blocks": [13, 14]},
                ),
            )

    def test_interrupted_save_never_publishes_a_partial_checkpoint(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "recovery"
            optimizer, scheduler = _optimizer_and_scheduler()
            complete = _recovery(
                next_source_block=8,
                optimization_window_index=1,
                completed_policy_steps=4,
            )
            save_recovery_checkpoint(
                _Bundle(), optimizer, scheduler, complete, root, _signature()
            )
            interrupted = _recovery(
                next_source_block=16,
                optimization_window_index=2,
                completed_policy_steps=8,
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
