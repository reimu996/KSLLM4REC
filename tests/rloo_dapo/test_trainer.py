from __future__ import annotations

from dataclasses import dataclass
import unittest

import torch

from ksllm4rec_rloo_dapo.config import approved_config
from ksllm4rec_rloo_dapo.trainer import (
    WindowLRScheduler,
    configure_deterministic_runtime,
    learning_rate_for_window,
    partition_effective_groups,
)


@dataclass(frozen=True)
class _Source:
    group_id: str


@dataclass(frozen=True)
class _Prepared:
    group: _Source


class WindowScheduleTest(unittest.TestCase):
    def setUp(self) -> None:
        self.config = approved_config()

    def test_linear_warmup_is_counted_in_complete_windows(self) -> None:
        expected = {
            0: 0.0,
            1: 1.0e-7,
            9: 9.0e-7,
            10: 1.0e-6,
            1063: 1.0e-6,
        }
        for window, value in expected.items():
            with self.subTest(window=window):
                self.assertAlmostEqual(
                    learning_rate_for_window(self.config, window), value, places=15
                )

    def test_deterministic_runtime_enables_flash_and_torch_controls(self) -> None:
        import os

        previous = torch.are_deterministic_algorithms_enabled()
        try:
            configure_deterministic_runtime(42)
            self.assertEqual(os.environ["FLASH_ATTENTION_DETERMINISTIC"], "1")
            self.assertEqual(os.environ["CUBLAS_WORKSPACE_CONFIG"], ":4096:8")
            self.assertTrue(torch.are_deterministic_algorithms_enabled())
            self.assertTrue(torch.backends.cudnn.deterministic)
            self.assertFalse(torch.backends.cudnn.benchmark)
        finally:
            torch.use_deterministic_algorithms(previous)

    def test_scheduler_steps_once_after_four_optimizer_updates(self) -> None:
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.SGD([parameter], lr=9.0)
        scheduler = WindowLRScheduler(optimizer, self.config)
        self.assertEqual(scheduler.completed_windows, 0)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)
        # The trainer may call optimizer.step four times; LR remains unchanged.
        for _ in range(4):
            optimizer.step()
            self.assertEqual(optimizer.param_groups[0]["lr"], 0.0)
        scheduler.step()
        self.assertEqual(scheduler.completed_windows, 1)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1.0e-7)


class MinibatchPartitionTest(unittest.TestCase):
    def test_every_group_occurs_once_across_four_minibatches(self) -> None:
        groups = tuple(_Prepared(_Source(f"g{index:02d}")) for index in range(32))
        batches = partition_effective_groups(groups, window_index=7, seed=42)
        self.assertEqual([len(batch) for batch in batches], [8, 8, 8, 8])
        ids = [item.group.group_id for batch in batches for item in batch]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), {item.group.group_id for item in groups})
        self.assertEqual(
            batches,
            partition_effective_groups(groups, window_index=7, seed=42),
        )


if __name__ == "__main__":
    unittest.main()
