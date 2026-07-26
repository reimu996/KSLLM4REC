from __future__ import annotations

import unittest

import torch

from ksllm4rec_dapo_anchor_v2_2.config import build_config
from ksllm4rec_dapo_anchor_v2_2.trainer import (
    WindowLRScheduler,
    learning_rate_for_window,
)


class WindowScheduleTest(unittest.TestCase):
    def test_anchor_starts_at_w0_while_lr_warmup_remains_ten_windows(self) -> None:
        config = build_config()
        self.assertEqual(config["anchor"]["warmup_windows"], 0)
        self.assertEqual(config["train"]["warmup_windows"], 10)
        self.assertEqual(config["train"]["max_total_optimizer_updates"], 5320)

    def test_lr_is_nonzero_at_w0_and_reaches_one_e_minus_six_at_w9(self) -> None:
        config = build_config()
        self.assertEqual(learning_rate_for_window(config, 0), 0.1e-6)
        self.assertEqual(learning_rate_for_window(config, 5), 0.6e-6)
        self.assertEqual(learning_rate_for_window(config, 9), 1.0e-6)
        self.assertEqual(learning_rate_for_window(config, 10), 1.0e-6)
        self.assertEqual(learning_rate_for_window(config, 100), 1.0e-6)

    def test_scheduler_advances_once_per_complete_window(self) -> None:
        config = build_config()
        parameter = torch.nn.Parameter(torch.tensor(1.0))
        optimizer = torch.optim.AdamW([parameter], lr=99.0)
        scheduler = WindowLRScheduler(optimizer, config, completed_windows=8)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.9e-6)
        scheduler.step()
        self.assertEqual(scheduler.completed_windows, 9)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1.0e-6)
        scheduler.step()
        self.assertEqual(scheduler.completed_windows, 10)
        self.assertEqual(optimizer.param_groups[0]["lr"], 1.0e-6)


if __name__ == "__main__":
    unittest.main()
