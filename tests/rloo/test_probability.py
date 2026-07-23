import hashlib
import unittest

import torch

from ksllm4rec_rloo.probability import (
    legal_log_probs,
    legal_token_log_prob,
    sample_legal_action,
    sample_legal_action_with_stats,
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

    def test_temperature_stats_use_the_sampling_distribution(self) -> None:
        logits = torch.tensor([4.0, 2.0, 0.0])
        generator = torch.Generator().manual_seed(7)
        _, _, decision, entropy, count = sample_legal_action_with_stats(
            logits, [0, 1, 2], generator=generator, temperature=1.2
        )
        expected_logps = torch.log_softmax(logits / 1.2, dim=0)
        expected_entropy = -(expected_logps.exp() * expected_logps).sum()
        self.assertTrue(decision)
        self.assertEqual(count, 3)
        torch.testing.assert_close(entropy, expected_entropy)

        logps_t1 = torch.log_softmax(logits, dim=0)
        entropy_t1 = -(logps_t1.exp() * logps_t1).sum()
        self.assertGreater(float(entropy), float(entropy_t1))

    def test_stats_entropy_handles_zero_probability_legal_action(self) -> None:
        logits = torch.tensor([0.0, -torch.inf])
        _, _, _, entropy, count = sample_legal_action_with_stats(logits, [0, 1])
        self.assertEqual(count, 2)
        self.assertTrue(torch.isfinite(entropy))
        self.assertEqual(float(entropy), 0.0)

    def test_stats_singleton_does_not_consume_generator(self) -> None:
        generator = torch.Generator().manual_seed(29)
        before = generator.get_state().clone()
        _, _, decision, entropy, count = sample_legal_action_with_stats(
            torch.tensor([1.0, 2.0]), [1], generator=generator, temperature=1.2
        )
        self.assertFalse(decision)
        self.assertEqual(count, 1)
        self.assertEqual(float(entropy), 0.0)
        self.assertTrue(torch.equal(before, generator.get_state()))

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
