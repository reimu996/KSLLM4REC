import unittest

import torch

from ksllm4rec_grpo.trainer import _pad_float_rows


class PaddingTest(unittest.TestCase):
    def test_forced_only_chunk_keeps_two_dimensional_old_log_probs(self):
        value = _pad_float_rows([], width=19, device=torch.device("cpu"))
        self.assertEqual(value.shape, (0, 19))
        self.assertEqual(value.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
