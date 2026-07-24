from __future__ import annotations

import unittest

import torch
from torch import nn

from ksllm4rec_orpo.data import Sid
from ksllm4rec_rloo_dapo.scoring import (
    grammar_completion_width,
    score_completions_dense_fixed,
    score_prefix_distribution_fixed,
)


class _Output:
    def __init__(self, hidden: torch.Tensor) -> None:
        self.last_hidden_state = hidden


class _Backbone(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(128, hidden_size)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> _Output:
        del kwargs
        return _Output(self.embedding(input_ids))


class _ToyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Backbone(4)
        self.lm_head = nn.Linear(4, 128, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask=None, **kwargs):
        del attention_mask, kwargs
        hidden = self.model(input_ids)
        return type("Out", (), {"logits": self.lm_head(hidden.last_hidden_state)})()


class _Grammar:
    eos_token_id = 99

    def allowed_next(self, prefix):
        if not prefix:
            return [10, 11]
        if prefix == [10]:
            return [12]
        if len(prefix) == 1 and prefix[0] == 11:
            return [13, 14]
        return [self.eos_token_id]

    def parse(self, tokens):
        if tokens not in ([10, 12, 99], [11, 13, 99], [11, 14, 99]):
            raise ValueError("invalid completion")
        return Sid("video", 1, 2, 3)


class DenseFixedScoringTest(unittest.TestCase):
    def test_runtime_width_is_derived_from_reachable_grammar_tokens(self) -> None:
        grammar = _Grammar()
        grammar.prefix_tokens = {"short": (1, 2), "long": (1, 2, 3, 4)}
        grammar.suffix_tokens = (99,)
        self.assertEqual(grammar_completion_width(grammar, safety_limit=12), 8)

    def test_one_forward_scores_all_decisions_and_retains_gradients(self) -> None:
        model = _ToyLM()
        result = score_completions_dense_fixed(
            model,
            [1, 2],
            [[10, 12, 99], [11, 13, 99]],
            _Grammar(),
            device="cpu",
            temperature=1.2,
            batch_rows=2,
            completion_width=4,
        )

        self.assertEqual(result.log_probs.shape, (2, 4))
        self.assertEqual(
            result.decision_mask.tolist(),
            [[True, False, False, False], [True, True, False, False]],
        )
        self.assertEqual([item.position for item in result.decisions[0]], [0])
        self.assertEqual([item.position for item in result.decisions[1]], [0, 1])
        self.assertTrue(result.log_probs.requires_grad)
        result.log_probs[result.decision_mask].sum().backward()
        self.assertIsNotNone(model.lm_head.weight.grad)

    def test_prefix_distribution_uses_the_same_fixed_shape(self) -> None:
        model = _ToyLM()
        grammar = _Grammar()
        dense = score_completions_dense_fixed(
            model,
            [1, 2],
            [[11, 13, 99], [10, 12, 99]],
            grammar,
            device="cpu",
            batch_rows=2,
            completion_width=4,
        )
        prefix = score_prefix_distribution_fixed(
            model,
            [1, 2],
            [11],
            grammar,
            device="cpu",
            batch_rows=2,
            completion_width=4,
        )

        self.assertEqual(prefix.allowed_ids, (13, 14))
        self.assertTrue(torch.equal(prefix.log_probs, dense.decisions[0][1].log_probs))


if __name__ == "__main__":
    unittest.main()
