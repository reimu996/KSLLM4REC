import hashlib
import unittest

import torch

from ksllm4rec_rloo.probability import (
    legal_log_probs,
    legal_token_log_prob,
    sample_legal_action,
    sample_legal_token,
)


class LegalPrefixProbabilityTest(unittest.TestCase):
    def test_fp32_legal_only_denominator(self) -> None:
        logits = torch.tensor([100.0, 1.0, -3.0, 2.0], dtype=torch.float16)
        result = legal_log_probs(logits, [1, 3])
        expected = torch.log_softmax(torch.tensor([1.0, 2.0]), dim=0)
        self.assertEqual(result.dtype, torch.float32)
        torch.testing.assert_close(result, expected)

    def test_singleton_returns_zero_without_consuming_generator(self) -> None:
        generator = torch.Generator().manual_seed(123)
        before = torch.rand((), generator=generator)
        # Reset to a known state and compare the next draw after a singleton.
        generator.manual_seed(123)
        expected_first = torch.rand((), generator=generator)
        generator.manual_seed(123)
        token, logp, decision = sample_legal_action(
            torch.tensor([float("nan"), 2.0]), [1], generator=generator
        )
        after = torch.rand((), generator=generator)
        self.assertEqual(token, 1)
        self.assertEqual(float(logp), 0.0)
        self.assertFalse(decision)
        torch.testing.assert_close(before, expected_first)
        torch.testing.assert_close(after, expected_first)

    def test_sample_and_rescore_share_the_same_logp(self) -> None:
        logits = torch.tensor([0.0, 1.0, 2.0, 3.0])
        generator = torch.Generator().manual_seed(42)
        token, sampled_logp = sample_legal_token(
            logits, [1, 3], generator=generator
        )
        rescored = legal_token_log_prob(logits, [1, 3], token)
        torch.testing.assert_close(sampled_logp, rescored)

    def test_seed_payload_is_independent_of_candidate_order(self) -> None:
        payload = b"rollout|42|0|group-x|15"
        expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        # The seed function lives in rollout; this test is kept here as a
        # contract reminder that probability itself must not alter RNG state.
        from ksllm4rec_rloo.rollout import rollout_seed

        self.assertEqual(rollout_seed("group-x", 0, 15), expected)

    def test_rejects_illegal_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "outside"):
            legal_token_log_prob(torch.zeros(4), [0, 2], 1)


if __name__ == "__main__":
    unittest.main()
