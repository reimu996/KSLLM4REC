from __future__ import annotations

import math
import unittest

import torch

from ksllm4rec_dapo_anchor_v2_2.objective import (
    anchor_effective_lambda,
    gt_sequence_logps,
    gt_set_anchor_loss,
)


class GtSetAnchorObjectiveTest(unittest.TestCase):
    def test_gt_set_loss_is_negative_log_total_probability(self) -> None:
        # Two distinct legal GT paths with probabilities 0.2 and 0.3.
        # The target is their total mass, not their mean token NLL.
        sequence_logps = torch.log(torch.tensor([0.2, 0.3]))
        loss = gt_set_anchor_loss(sequence_logps)
        self.assertAlmostEqual(float(loss), -math.log(0.5), places=6)

    def test_sequence_logps_only_sum_grammar_decision_tokens(self) -> None:
        logps = torch.tensor(
            [
                [-0.7, 0.0, 0.0, 0.0],
                [-0.2, -0.4, 0.0, 0.0],
            ]
        )
        mask = torch.tensor(
            [
                [True, False, False, False],
                [True, True, False, False],
            ]
        )
        sequence, counts = gt_sequence_logps(logps, mask)
        torch.testing.assert_close(sequence, torch.tensor([-0.7, -0.6]))
        self.assertEqual(counts.tolist(), [1, 2])


class AnchorGradientBudgetTest(unittest.TestCase):
    def test_runtime_cap_keeps_anchor_at_ten_percent_of_rl_reference(self) -> None:
        # lambda_calibrated alone is too large.  The runtime cap must win.
        result = anchor_effective_lambda(
            lambda_calibrated=0.01,
            rl_reference_grad_norm=0.05,
            anchor_raw_grad_norm=4.0,
            target_gradient_ratio=0.10,
        )
        self.assertAlmostEqual(result.lambda_cap, 0.00125, places=12)
        self.assertAlmostEqual(result.lambda_effective, 0.00125, places=12)
        self.assertAlmostEqual(result.anchor_to_rl_grad_ratio, 0.10, places=12)

    def test_runtime_cap_never_increases_calibrated_weight(self) -> None:
        result = anchor_effective_lambda(
            lambda_calibrated=0.0001,
            rl_reference_grad_norm=1.0,
            anchor_raw_grad_norm=1.0,
            target_gradient_ratio=0.10,
        )
        self.assertAlmostEqual(result.lambda_effective, 0.0001, places=12)
        self.assertAlmostEqual(result.anchor_to_rl_grad_ratio, 0.0001, places=12)

    def test_zero_reference_or_zero_anchor_skips_anchor_update(self) -> None:
        for rl_norm, anchor_norm in ((0.0, 3.0), (0.1, 0.0)):
            with self.subTest(rl_norm=rl_norm, anchor_norm=anchor_norm):
                result = anchor_effective_lambda(
                    lambda_calibrated=0.01,
                    rl_reference_grad_norm=rl_norm,
                    anchor_raw_grad_norm=anchor_norm,
                    target_gradient_ratio=0.10,
                )
                self.assertEqual(result.lambda_effective, 0.0)
                self.assertEqual(result.anchor_to_rl_grad_ratio, 0.0)


if __name__ == "__main__":
    unittest.main()
