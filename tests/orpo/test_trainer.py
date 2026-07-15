from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from ksllm4rec_orpo.trainer import ChunkedORPOTrainer


class FakeAccelerator:
    @staticmethod
    def unwrap_model(model):
        return model


class FakeBackbone(nn.Module):
    def forward(self, input_ids, **kwargs):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 4)
        hidden.requires_grad_(True)
        return SimpleNamespace(last_hidden_state=hidden)


class FakeCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeBackbone()
        self.lm_head = nn.Linear(4, 16, bias=False)
        self.lm_head.weight.requires_grad_(False)
        self.head_lengths: list[int] = []

    def forward(self, input_ids, logits_to_keep=0, **kwargs):
        output = self.model(input_ids=input_ids, **kwargs)
        selected = output.last_hidden_state[:, -logits_to_keep:, :]
        self.head_lengths.append(selected.size(1))
        return SimpleNamespace(logits=self.lm_head(selected))


class ChunkedORPOTrainerForwardTest(unittest.TestCase):
    def test_pair_forward_materializes_only_placeholder_logits(self) -> None:
        torch.manual_seed(42)
        trainer = object.__new__(ChunkedORPOTrainer)
        trainer.accelerator = FakeAccelerator()
        trainer.lm_chunk_size = 2
        trainer.beta = 0.1
        trainer.micro_step = 0
        trainer.min_sequence_length = None
        trainer.max_sequence_length = None
        trainer.min_chosen_tokens = None
        trainer.max_chosen_tokens = None
        trainer.min_rejected_tokens = None
        trainer.max_rejected_tokens = None
        model = FakeCausalLM()
        input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6],
                [1, 2, 3, 7, 8, 9],
                [1, 2, 3, 4, 5, 10],
                [1, 2, 3, 7, 8, 11],
            ]
        )
        labels = torch.tensor(
            [
                [-100, -100, -100, 4, 5, 6],
                [-100, -100, -100, 7, 8, 9],
                [-100, -100, -100, 4, 5, 10],
                [-100, -100, -100, 7, 8, 11],
            ]
        )
        loss, metrics = trainer.get_batch_loss_metrics(
            model,
            {
                "input_ids": input_ids,
                "attention_mask": torch.ones_like(input_ids),
                "labels": labels,
            },
        )
        loss.backward()
        self.assertEqual(model.head_lengths, [1])
        self.assertEqual(trainer.micro_step, 1)
        self.assertEqual(trainer.min_sequence_length, 6)
        self.assertEqual(trainer.min_chosen_tokens, 3)
        self.assertIn("odds_ratio_loss", metrics)
        self.assertTrue(bool(torch.isfinite(loss)))


if __name__ == "__main__":
    unittest.main()
