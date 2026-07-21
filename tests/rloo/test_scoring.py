import unittest

import torch
from torch import nn

from ksllm4rec_orpo.data import Sid
from ksllm4rec_rloo.scoring import (
    completion_prediction_hidden,
    score_completions,
    score_completions_chunked,
)


class _Output:
    def __init__(self, hidden):
        self.last_hidden_state = hidden


class _Backbone(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.embedding = nn.Embedding(128, hidden)

    def forward(self, input_ids, **kwargs):
        del kwargs
        return _Output(self.embedding(input_ids))


class _ToyLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = _Backbone(4)
        self.lm_head = nn.Linear(4, 128, bias=False)

    def forward(self, input_ids, attention_mask=None, **kwargs):
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


class ScoringTest(unittest.TestCase):
    def test_hidden_alignment(self):
        hidden = torch.arange(2 * 9, dtype=torch.float32).reshape(2, 9, 1)
        selected = completion_prediction_hidden(
            hidden, prompt_length=4, completion_length=3
        )
        self.assertEqual(selected.shape, (2, 3, 1))
        self.assertEqual(selected[0, :, 0].tolist(), [3.0, 4.0, 5.0])

    def test_scores_are_padded_and_singleton_nodes_have_zero_logp(self):
        result = score_completions(
            _ToyLM(),
            [1, 2],
            [[10, 12, 99], [11, 13, 99]],
            _Grammar(),
            device="cpu",
        )
        self.assertEqual(result.log_probs.shape, (2, 3))
        self.assertEqual(result.valid_mask.tolist(), [[True] * 3, [True] * 3])
        # First row: token 12 and EOS are deterministic after selecting 10.
        self.assertEqual(float(result.log_probs[0, 1]), 0.0)
        self.assertEqual(float(result.log_probs[0, 2]), 0.0)
        self.assertEqual(result.decision_mask.tolist(), [[True, False, False], [True, True, False]])

    def test_chunked_gt_scoring_does_not_aggregate_chunks(self):
        result = score_completions_chunked(
            _ToyLM(),
            [1, 2],
            [[10, 12, 99], [11, 13, 99], [11, 14, 99]],
            _Grammar(),
            chunk_size=2,
            device="cpu",
        )
        self.assertEqual(len(result), 2)
        self.assertEqual([chunk.log_probs.size(0) for chunk in result], [2, 1])

    def test_illegal_completion_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "illegal"):
            score_completions(
                _ToyLM(), [1], [[10, 99]], _Grammar(), device="cpu"
            )


if __name__ == "__main__":
    unittest.main()
