from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
from torch import nn

from ksllm4rec_sft.trainer import _forward_with_captured_hidden


class FakeBackbone(nn.Module):
    def forward(self, input_ids, **kwargs):
        hidden = input_ids.float().unsqueeze(-1).repeat(1, 1, 3)
        hidden.requires_grad_(True)
        return SimpleNamespace(last_hidden_state=hidden)


class FakeCausalLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = FakeBackbone()
        self.lm_head = nn.Linear(3, 5, bias=False)
        self.head_sequence_lengths: list[int] = []

    def forward(self, input_ids, logits_to_keep=0, **kwargs):
        backbone_output = self.model(input_ids=input_ids, **kwargs)
        selected = backbone_output.last_hidden_state[:, -logits_to_keep:, :]
        self.head_sequence_lengths.append(selected.size(1))
        return SimpleNamespace(logits=self.lm_head(selected))


class FakeWrapper(nn.Module):
    def __init__(self, module: nn.Module) -> None:
        super().__init__()
        self.module = module
        self.calls = 0

    def forward(self, **kwargs):
        self.calls += 1
        return self.module(**kwargs)


class CapturedBackboneForwardTest(unittest.TestCase):
    def test_uses_wrapper_and_only_one_placeholder_logit_position(self) -> None:
        causal_lm = FakeCausalLM()
        wrapper = FakeWrapper(causal_lm)
        hidden, outputs = _forward_with_captured_hidden(
            wrapper,
            causal_lm,
            {
                "input_ids": torch.tensor([[1, 2, 3, 4]]),
                "logits_to_keep": 1,
                "use_cache": False,
                "return_dict": True,
            },
        )
        self.assertEqual(wrapper.calls, 1)
        self.assertEqual(causal_lm.head_sequence_lengths, [1])
        self.assertEqual(tuple(hidden.shape), (1, 4, 3))
        self.assertEqual(tuple(outputs.logits.shape), (1, 1, 5))
        hidden.sum().backward()
        self.assertIsNotNone(hidden.grad)


if __name__ == "__main__":
    unittest.main()
