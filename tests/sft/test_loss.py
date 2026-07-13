from __future__ import annotations

import unittest

import torch
import torch.nn.functional as F

from ksllm4rec_sft.loss import (
    chunked_focal_loss,
    reference_focal_loss,
    summarize_metrics,
)


class ChunkedFocalLossTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.hidden = torch.randn(2, 7, 5, dtype=torch.float32)
        self.weight = torch.randn(11, 5, dtype=torch.float32)
        self.labels = torch.tensor(
            [[-100, -100, 2, 3, 4, -100, 5], [-100, 6, 7, -100, 8, 9, 10]],
            dtype=torch.long,
        )
        self.item_mask = torch.tensor(
            [
                [False, False, True, False, True, False, False],
                [False, True, False, False, True, False, True],
            ]
        )

    def test_loss_and_hidden_gradient_match_reference(self) -> None:
        reference_hidden = self.hidden.clone().requires_grad_(True)
        logits = F.linear(reference_hidden, self.weight)
        reference_loss, reference_ce = reference_focal_loss(
            logits, self.labels, self.item_mask
        )
        reference_loss.backward()

        for chunk_size in (1, 2, 512):
            with self.subTest(chunk_size=chunk_size):
                test_hidden = self.hidden.clone().requires_grad_(True)
                test_loss, test_ce, test_items = chunked_focal_loss(
                    test_hidden,
                    self.weight,
                    self.labels,
                    self.item_mask,
                    chunk_size=chunk_size,
                )
                test_loss.backward()
                self.assertLessEqual(
                    abs(test_loss.item() - reference_loss.item()), 1e-6
                )
                torch.testing.assert_close(test_ce, reference_ce, atol=2e-6, rtol=0.0)
                torch.testing.assert_close(
                    test_hidden.grad, reference_hidden.grad, atol=1e-5, rtol=0.0
                )
                self.assertEqual(test_items.numel(), reference_ce.numel())

    def test_metrics_match_approved_numeric_example(self) -> None:
        ce = torch.tensor([0.1, 1.0, 2.0, 0.5])
        items = torch.tensor([False, True, False, True])
        metrics = summarize_metrics(ce, items)
        self.assertAlmostEqual(metrics.item_ratio, 0.5)
        self.assertAlmostEqual(metrics.item_loss, 0.75)
        self.assertAlmostEqual(metrics.text_loss, 1.05)
        self.assertAlmostEqual(metrics.dataset_loss, 0.9)

    def test_empty_valid_set_is_rejected(self) -> None:
        labels = torch.full((1, 3), -100, dtype=torch.long)
        with self.assertRaises(ValueError):
            chunked_focal_loss(
                torch.randn(1, 3, 2),
                torch.randn(4, 2),
                labels,
                torch.zeros_like(labels, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
