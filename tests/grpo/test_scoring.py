import unittest

import torch

from ksllm4rec_grpo.scoring import completion_prediction_hidden


class CompletionAlignmentTest(unittest.TestCase):
    def test_first_completion_and_last_eos_use_exact_predecessor_positions(self):
        hidden = torch.arange(2 * 9 * 1, dtype=torch.float32).reshape(2, 9, 1)
        selected = completion_prediction_hidden(
            hidden, prompt_length=4, completion_length=3
        )
        self.assertEqual(selected.shape, (2, 3, 1))
        self.assertEqual(selected[0, :, 0].tolist(), [3.0, 4.0, 5.0])
        self.assertEqual(selected[1, :, 0].tolist(), [12.0, 13.0, 14.0])

    def test_rejects_slice_past_hidden_sequence(self):
        with self.assertRaisesRegex(ValueError, "too short"):
            completion_prediction_hidden(
                torch.zeros(1, 4, 2), prompt_length=4, completion_length=2
            )


if __name__ == "__main__":
    unittest.main()
