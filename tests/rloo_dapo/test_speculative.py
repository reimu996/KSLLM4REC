from __future__ import annotations

import math
import unittest

import torch

from ksllm4rec_rloo_dapo.speculative import speculative_accept_or_residual


class SpeculativeCorrectionTest(unittest.TestCase):
    def test_accepts_the_proposal_under_the_probability_ratio(self) -> None:
        result = speculative_accept_or_residual(
            torch.log(torch.tensor([0.25, 0.75])),
            torch.log(torch.tensor([0.50, 0.50])),
            proposal_index=0,
            acceptance_uniform=0.49,
            residual_uniform=0.0,
        )
        self.assertTrue(result.accepted)
        self.assertEqual(result.selected_index, 0)
        self.assertTrue(math.isclose(result.acceptance_probability, 1.0))

    def test_rejection_samples_only_from_positive_target_residual(self) -> None:
        result = speculative_accept_or_residual(
            torch.log(torch.tensor([0.90, 0.10])),
            torch.log(torch.tensor([0.10, 0.90])),
            proposal_index=0,
            acceptance_uniform=0.5,
            residual_uniform=0.25,
        )
        self.assertFalse(result.accepted)
        self.assertEqual(result.selected_index, 1)
        self.assertTrue(
            math.isclose(result.acceptance_probability, 1.0 / 9.0, rel_tol=1e-6)
        )

    def test_invalid_probability_vectors_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "same shape"):
            speculative_accept_or_residual(
                torch.zeros(2),
                torch.zeros(3),
                proposal_index=0,
                acceptance_uniform=0.0,
                residual_uniform=0.0,
            )


if __name__ == "__main__":
    unittest.main()
