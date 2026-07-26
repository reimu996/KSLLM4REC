from __future__ import annotations

import unittest

import torch
from torch import nn

from ksllm4rec_dapo_anchor_v2_2._infra.dapo_scoring import (
    score_completions_dense_fixed,
)
from ksllm4rec_dapo_anchor_v2_2.anchor_scoring import (
    score_prompt_completion_pairs_dense,
)


class _Output:
    def __init__(self, hidden: torch.Tensor) -> None:
        self.last_hidden_state = hidden


class _Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(128, 4)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> _Output:
        del kwargs
        return _Output(self.embedding(input_ids))


class _ToyLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Backbone()
        self.lm_head = nn.Linear(4, 128, bias=False)


class _Grammar:
    eos_token_id = 99

    def allowed_next(self, prefix):
        if not prefix:
            return [10, 11]
        if prefix == [10]:
            return [12]
        if prefix == [11]:
            return [13, 14]
        return [self.eos_token_id]

    def parse(self, tokens):
        if tokens not in ([10, 12, 99], [11, 13, 99]):
            raise ValueError("invalid completion")
        return tuple(tokens)


class DenseAnchorScoringTest(unittest.TestCase):
    def test_dense_no_cache_scores_decisions_and_retains_gradient(self) -> None:
        model = _ToyLM()
        scores = score_completions_dense_fixed(
            model,
            [1, 2],
            [[10, 12, 99], [11, 13, 99]],
            _Grammar(),
            device="cpu",
            temperature=1.0,
            batch_rows=2,
            completion_width=4,
        )
        self.assertEqual(
            scores.decision_mask.tolist(),
            [[True, False, False, False], [True, True, False, False]],
        )
        scores.log_probs[scores.decision_mask].sum().backward()
        self.assertIsNotNone(model.lm_head.weight.grad)

    def test_heterogeneous_batch_matches_existing_dense_scorer(self) -> None:
        torch.manual_seed(7)
        model = _ToyLM()
        grammar = _Grammar()
        completions = [[10, 12, 99], [11, 13, 99]]
        existing = score_completions_dense_fixed(
            model,
            [1, 2],
            completions,
            grammar,
            device="cpu",
            temperature=1.0,
            batch_rows=2,
            completion_width=4,
        )
        heterogeneous = score_prompt_completion_pairs_dense(
            model,
            [[1, 2], [1, 2]],
            completions,
            grammar,
            device="cpu",
            temperature=1.0,
            completion_width=4,
        )
        torch.testing.assert_close(heterogeneous.log_probs, existing.log_probs)
        self.assertTrue(
            torch.equal(heterogeneous.decision_mask, existing.decision_mask)
        )


if __name__ == "__main__":
    unittest.main()
