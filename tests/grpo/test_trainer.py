import unittest
from pathlib import Path

import torch

from ksllm4rec_grpo.config import load_config
from ksllm4rec_grpo.trainer import _pad_float_rows, training_schedule


class PaddingTest(unittest.TestCase):
    def test_forced_only_chunk_keeps_two_dimensional_old_log_probs(self):
        value = _pad_float_rows([], width=19, device=torch.device("cpu"))
        self.assertEqual(value.shape, (0, 19))
        self.assertEqual(value.dtype, torch.float32)


class TrainingScheduleTest(unittest.TestCase):
    def test_historical_schedule_is_unchanged(self):
        config = load_config(Path("configs/grpo/onereason_lora_grpo.yaml"))
        self.assertEqual(training_schedule(config, 6_378), (1_596, 48))

    def test_frontier_schedule_is_derived_from_group_count(self):
        config = load_config(Path("configs/grpo/frontier_sft_epoch2_grpo.yaml"))
        self.assertEqual(training_schedule(config, 17_016), (4_254, 128))


if __name__ == "__main__":
    unittest.main()
