from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from ksllm4rec_dapo_anchor_multitask_v1_1.anchor_scoring import (
    score_prompt_completion_pairs_dense,
)
from ksllm4rec_dapo_anchor_multitask_v1_1.config import build_config
from ksllm4rec_dapo_anchor_multitask_v1_1.data import SidTask
from ksllm4rec_dapo_anchor_multitask_v1_1._infra.orpo_data import Sid
from ksllm4rec_dapo_anchor_multitask_v1_1.objective import (
    gt_sequence_logps,
    gt_set_anchor_loss,
)
from ksllm4rec_dapo_anchor_multitask_v1_1.training import (
    build_anchor_raw_gradient,
)


class _Output:
    def __init__(self, hidden: torch.Tensor) -> None:
        self.last_hidden_state = hidden


class _Backbone(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = torch.nn.Embedding(128, 4)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> _Output:
        del kwargs
        return _Output(self.embedding(input_ids))


class _ToyLM(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _Backbone()
        self.lm_head = torch.nn.Linear(4, 128, bias=False)


class _Grammar:
    eos_token_id = 99

    def __init__(self, first: int) -> None:
        self.first = first

    def allowed_next(self, prefix):
        if not prefix:
            return [self.first, self.first + 1]
        if prefix == [self.first]:
            return [12]
        if prefix == [self.first + 1]:
            return [13, 14]
        return [99]

    def parse(self, tokens):
        if tokens not in (
            [self.first, 12, 99],
            [self.first + 1, 13, 99],
        ):
            raise ValueError("invalid")
        return tokens

    def encode_sid(self, sid: Sid):
        return [self.first, 12, 99] if sid.c == 0 else [self.first + 1, 13, 99]


class MixedTaskAnchorTest(unittest.TestCase):
    @patch("torch.cuda.is_available", return_value=False)
    def test_two_task_anchor_equals_direct_mean_gt_set_gradient(
        self, _cuda_available
    ) -> None:
        torch.manual_seed(9)
        actual_model = _ToyLM()
        expected_model = _ToyLM()
        expected_model.load_state_dict(actual_model.state_dict())
        rec_grammar = _Grammar(10)
        text_grammar = _Grammar(20)
        sid0 = Sid("video", 1, 1, 0)
        sid1 = Sid("video", 1, 1, 1)
        groups = (
            SimpleNamespace(
                identity=(0, SidTask.RECOMMENDATION.value, "rec"),
                epoch_index=0,
                source_block=3,
                epoch_block_index=3,
                group=SimpleNamespace(
                    task=SidTask.RECOMMENDATION, group_id="rec"
                ),
                prompt_ids=(1, 2),
                positives=(sid0, sid1),
            ),
            SimpleNamespace(
                identity=(1, SidTask.ITEM_TEXT_TO_SID.value, "text"),
                epoch_index=1,
                source_block=535,
                epoch_block_index=3,
                group=SimpleNamespace(
                    task=SidTask.ITEM_TEXT_TO_SID, group_id="text"
                ),
                prompt_ids=(3, 4, 5),
                positives=(sid1,),
            ),
        )
        grammars = {
            SidTask.RECOMMENDATION: rec_grammar,
            SidTask.ITEM_TEXT_TO_SID: text_grammar,
        }
        config = build_config()
        optimizer = torch.optim.AdamW(actual_model.parameters(), lr=0.0)
        result = build_anchor_raw_gradient(
            SimpleNamespace(model=actual_model),
            optimizer,
            groups,
            grammars,
            config=config,
            device="cpu",
        )

        expected_losses = []
        for item in groups:
            grammar = grammars[item.group.task]
            completions = [grammar.encode_sid(sid) for sid in item.positives]
            scores = score_prompt_completion_pairs_dense(
                expected_model,
                [item.prompt_ids] * len(completions),
                completions,
                grammar,
                device="cpu",
                temperature=1.0,
                completion_width=32,
            )
            sequences, _ = gt_sequence_logps(
                scores.log_probs, scores.decision_mask
            )
            expected_losses.append(gt_set_anchor_loss(sequences))
        torch.stack(expected_losses).mean().backward()

        self.assertEqual(len(result.group_results), 2)
        self.assertEqual([item.gt_sid_count for item in result.group_results], [2, 1])
        self.assertEqual(
            [(item.epoch_index, item.source_block, item.epoch_block_index) for item in result.group_results],
            [(0, 3, 3), (1, 535, 3)],
        )
        self.assertEqual(
            [(item.task, item.group_id) for item in result.group_results],
            [("recommendation", "rec"), ("item_text_to_sid", "text")],
        )
        self.assertLessEqual(result.max_replay_logp_difference, 1.0e-5)
        for actual, expected in zip(
            actual_model.parameters(), expected_model.parameters(), strict=True
        ):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=1e-5, atol=1e-7)


if __name__ == "__main__":
    unittest.main()
