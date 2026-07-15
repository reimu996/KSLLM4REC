from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from ksllm4rec_orpo.loss import (
    chunked_sequence_logps,
    reference_sequence_logps,
    stable_orpo_loss,
)


class ChunkedSequenceLogpsTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(20260715)
        self.hidden = torch.randn(4, 8, 6, dtype=torch.float32)
        self.weight = torch.randn(13, 6, dtype=torch.float32)
        self.labels = torch.tensor(
            [
                [-100, -100, 1, 2, 3, -100, 4, 5],
                [-100, 2, 3, -100, 4, 5, 6, 7],
                [-100, -100, -100, 8, 9, 10, 11, 12],
                [-100, 1, -100, 2, -100, 3, -100, 4],
            ],
            dtype=torch.long,
        )

    def test_logps_loss_and_hidden_gradient_match_full_logits(self) -> None:
        reference_hidden = self.hidden.clone().requires_grad_(True)
        reference_logits = F.linear(reference_hidden, self.weight)
        reference_logps, reference_counts = reference_sequence_logps(
            reference_logits, self.labels
        )
        reference_loss = stable_orpo_loss(
            reference_logps[:2], reference_logps[2:], beta=0.1
        ).loss
        reference_loss.backward()

        for chunk_size in (1, 3, 512):
            with self.subTest(chunk_size=chunk_size):
                hidden = self.hidden.clone().requires_grad_(True)
                logps, counts = chunked_sequence_logps(
                    hidden,
                    self.weight,
                    self.labels,
                    chunk_size=chunk_size,
                )
                output = stable_orpo_loss(logps[:2], logps[2:], beta=0.1)
                output.loss.backward()
                torch.testing.assert_close(
                    logps, reference_logps, atol=2e-6, rtol=0.0
                )
                torch.testing.assert_close(counts, reference_counts)
                torch.testing.assert_close(
                    output.loss, reference_loss, atol=2e-6, rtol=0.0
                )
                torch.testing.assert_close(
                    hidden.grad,
                    reference_hidden.grad,
                    atol=2e-5,
                    rtol=0.0,
                )

    def test_stable_objective_matches_literal_formula(self) -> None:
        chosen = torch.tensor([-2.0, -0.8], requires_grad=True)
        rejected = torch.tensor([-1.2, -1.7], requires_grad=True)
        output = stable_orpo_loss(chosen, rejected, beta=0.1)
        literal_log_odds = (chosen - rejected) - (
            torch.log1p(-torch.exp(chosen)) - torch.log1p(-torch.exp(rejected))
        )
        literal = (
            -chosen + 0.1 * -F.logsigmoid(literal_log_odds)
        ).mean()
        torch.testing.assert_close(output.loss, literal)

    def test_near_zero_and_very_negative_logps_remain_finite(self) -> None:
        chosen = torch.tensor([-1.0e-12, -100.0], requires_grad=True)
        rejected = torch.tensor([-1.0e-8, -120.0], requires_grad=True)
        output = stable_orpo_loss(chosen, rejected)
        output.loss.backward()
        self.assertTrue(bool(torch.isfinite(output.loss)))
        self.assertTrue(bool(torch.isfinite(chosen.grad).all()))
        self.assertTrue(bool(torch.isfinite(rejected.grad).all()))

    def test_rejects_empty_response_row(self) -> None:
        labels = torch.tensor([[-100, -100, -100], [-100, 1, 2]])
        with self.assertRaisesRegex(ValueError, "Every response row"):
            chunked_sequence_logps(
                torch.randn(2, 3, 4), torch.randn(5, 4), labels
            )


if __name__ == "__main__":
    unittest.main()
