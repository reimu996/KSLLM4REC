from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import patch

import torch

from ksllm4rec_dapo_anchor_multitask_v1_1.recovery_audit import (
    compare_recovery_trajectories,
)


_ARTIFACTS = (
    "source_epoch_plan.json",
    "source_groups.jsonl",
    "groups.jsonl",
    "windows.jsonl",
    "anchor_groups.jsonl",
    "epoch_auxiliary_flushes.jsonl",
)


@dataclass(frozen=True)
class _State:
    next_source_block: int
    epoch_index: int
    completed_policy_steps: int
    total_optimizer_steps: int
    tensor: torch.Tensor


class RecoveryAuditComparisonTest(unittest.TestCase):
    def _trajectory(
        self,
        root: Path,
        *,
        suffix: str = "",
        adapter_config: dict[str, object] | None = None,
    ) -> tuple[Path, object, dict]:
        root.mkdir(parents=True)
        for relative in _ARTIFACTS:
            (root / relative).write_text(
                json.dumps({"artifact": relative}) + "\n", encoding="utf-8"
            )
        (root / "run_summary.json").write_text(
            json.dumps(
                {
                    "complete": False,
                    "completed_source_blocks": 4,
                    "last_checkpoint": str(root / "recovery" / suffix),
                }
            )
            + "\n",
            encoding="utf-8",
        )
        checkpoint = root / "recovery" / "checkpoint-block-000004-window-000001-update-000001"
        checkpoint.mkdir(parents=True)
        (checkpoint / "adapter_model.safetensors").write_bytes(b"adapter")
        if adapter_config is None:
            adapter_config = {
                "r": 64,
                "lora_alpha": 64,
                "target_modules": ["q_proj", "v_proj"],
            }
        (checkpoint / "adapter_config.json").write_text(
            json.dumps(adapter_config) + "\n", encoding="utf-8"
        )
        state = _State(4, 0, 1, 1, torch.tensor([1.0, 2.0]))
        training = {
            "optimizer": {"step": torch.tensor(1)},
            "scheduler": {"completed_policy_steps": 1},
            "rng": {"python": (3, 4)},
        }
        return checkpoint, state, training

    def test_identical_trajectories_pass_after_local_checkpoint_path_normalization(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_checkpoint, left_state, left_training = self._trajectory(
                root / "continuous", suffix="left"
            )
            right_checkpoint, right_state, right_training = self._trajectory(
                root / "recovered", suffix="right"
            )
            with patch(
                "ksllm4rec_dapo_anchor_multitask_v1_1.recovery_audit.load_latest_recovery",
                side_effect=(
                    (left_checkpoint, left_state, left_training),
                    (right_checkpoint, right_state, right_training),
                ),
            ):
                report = compare_recovery_trajectories(
                    root / "continuous", root / "recovered", {"sha256": "test"}
                )
            self.assertTrue(report["passed"])
            self.assertEqual(report["completed_source_blocks"], 4)
            self.assertEqual(report["completed_policy_steps"], 1)

    def test_target_module_order_does_not_change_recovery_semantics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_checkpoint, left_state, left_training = self._trajectory(
                root / "continuous",
                adapter_config={
                    "r": 64,
                    "lora_alpha": 64,
                    "target_modules": ["k_proj", "q_proj", "v_proj"],
                },
            )
            right_checkpoint, right_state, right_training = self._trajectory(
                root / "recovered",
                adapter_config={
                    "r": 64,
                    "lora_alpha": 64,
                    "target_modules": ["v_proj", "k_proj", "q_proj"],
                },
            )
            with patch(
                "ksllm4rec_dapo_anchor_multitask_v1_1.recovery_audit.load_latest_recovery",
                side_effect=(
                    (left_checkpoint, left_state, left_training),
                    (right_checkpoint, right_state, right_training),
                ),
            ):
                report = compare_recovery_trajectories(
                    root / "continuous", root / "recovered", {"sha256": "test"}
                )
            self.assertTrue(report["passed"])
            self.assertFalse(report["adapter_config"]["raw_byte_identical"])

    def test_target_module_set_difference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_checkpoint, left_state, left_training = self._trajectory(
                root / "continuous",
                adapter_config={"target_modules": ["q_proj", "v_proj"]},
            )
            right_checkpoint, right_state, right_training = self._trajectory(
                root / "recovered",
                adapter_config={"target_modules": ["q_proj", "o_proj"]},
            )
            with patch(
                "ksllm4rec_dapo_anchor_multitask_v1_1.recovery_audit.load_latest_recovery",
                side_effect=(
                    (left_checkpoint, left_state, left_training),
                    (right_checkpoint, right_state, right_training),
                ),
            ), self.assertRaisesRegex(AssertionError, "configuration differs semantically"):
                compare_recovery_trajectories(
                    root / "continuous", root / "recovered", {"sha256": "test"}
                )

    def test_empty_epoch_flush_log_is_a_valid_byte_identical_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_checkpoint, left_state, left_training = self._trajectory(
                root / "continuous", suffix="left"
            )
            right_checkpoint, right_state, right_training = self._trajectory(
                root / "recovered", suffix="right"
            )
            (root / "continuous" / "epoch_auxiliary_flushes.jsonl").write_bytes(b"")
            (root / "recovered" / "epoch_auxiliary_flushes.jsonl").write_bytes(b"")
            with patch(
                "ksllm4rec_dapo_anchor_multitask_v1_1.recovery_audit.load_latest_recovery",
                side_effect=(
                    (left_checkpoint, left_state, left_training),
                    (right_checkpoint, right_state, right_training),
                ),
            ):
                report = compare_recovery_trajectories(
                    root / "continuous", root / "recovered", {"sha256": "test"}
                )
            self.assertTrue(report["passed"])
            self.assertIn("epoch_auxiliary_flushes.jsonl", report["artifact_sha256"])

    def test_one_uncommitted_log_difference_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            left_checkpoint, left_state, left_training = self._trajectory(
                root / "continuous"
            )
            right_checkpoint, right_state, right_training = self._trajectory(
                root / "recovered"
            )
            (root / "recovered" / "groups.jsonl").write_text(
                '{"changed":true}\n', encoding="utf-8"
            )
            with patch(
                "ksllm4rec_dapo_anchor_multitask_v1_1.recovery_audit.load_latest_recovery",
                side_effect=(
                    (left_checkpoint, left_state, left_training),
                    (right_checkpoint, right_state, right_training),
                ),
            ), self.assertRaisesRegex(AssertionError, "groups.jsonl"):
                compare_recovery_trajectories(
                    root / "continuous", root / "recovered", {"sha256": "test"}
                )


if __name__ == "__main__":
    unittest.main()
