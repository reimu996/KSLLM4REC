import unittest

import torch

from ksllm4rec_grpo.probability import (
    legal_log_probs,
    legal_token_log_prob,
    sample_legal_token,
)


class LegalProbabilityTest(unittest.TestCase):
    def test_normalizes_only_over_legal_ids_in_fp32(self):
        logits = torch.tensor([100.0, 1.0, -3.0, 2.0], dtype=torch.float16)
        result = legal_log_probs(logits, [1, 3])
        expected = torch.log_softmax(torch.tensor([1.0, 2.0]), dim=0)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, expected)
        torch.testing.assert_close(result.exp().sum(), torch.tensor(1.0))

    def test_selected_token_uses_same_denominator(self):
        logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
        result = legal_token_log_prob(logits, [0, 2, 3], 2)
        self.assertEqual(result, legal_log_probs(logits, [0, 2, 3])[1])

    def test_illegal_target_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "outside"):
            legal_token_log_prob(torch.zeros(4), [0, 2], 1)

    def test_sampling_is_seed_reproducible_and_returns_its_logp(self):
        logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
        first = torch.Generator().manual_seed(42)
        second = torch.Generator().manual_seed(42)
        token_a, logp_a = sample_legal_token(logits, [1, 3], generator=first)
        token_b, logp_b = sample_legal_token(logits, [1, 3], generator=second)
        self.assertEqual(token_a, token_b)
        self.assertEqual(logp_a, logp_b)
        self.assertEqual(logp_a, legal_token_log_prob(logits, [1, 3], token_a))


if __name__ == "__main__":
    unittest.main()
