import hashlib
import unittest
from unittest.mock import patch

import torch

from ksllm4rec_orpo.data import Sid
from ksllm4rec_rloo.rollout import (
    RolloutCandidate,
    pad_rollout_candidates,
    rollout_group,
    rollout_seed,
)


class _Grammar:
    eos_token_id = 99

    def parse(self, tokens):
        if not tokens or tokens[-1] != self.eos_token_id:
            raise ValueError("missing eos")
        return Sid("video", 1, 2, 3)


class _Model:
    training = True

    def eval(self):
        self.training = False

    def train(self, value=True):
        self.training = bool(value)


def _fake_chunk(
    model,
    prompt_ids,
    grammar,
    *,
    group_id,
    epoch_index,
    candidate_indices,
    max_completion_length,
    device,
    temperature=1.0,
):
    del model, prompt_ids, grammar, group_id, epoch_index, max_completion_length
    del device, temperature
    return [
        RolloutCandidate(
            candidate_index=index,
            sid=Sid("video", 1, 2, 3),
            token_ids=(10, 99),
            old_log_probs=(0.0, 0.0),
            decision_mask=(True, False),
        )
        for index in candidate_indices
    ]


class G16RolloutTest(unittest.TestCase):
    def test_seed_matches_frozen_payload_and_all_16_are_unique(self) -> None:
        payload = b"rollout|42|2|group-abc|7"
        expected = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")
        self.assertEqual(rollout_seed("group-abc", 2, 7), expected)
        self.assertEqual(len({rollout_seed("g", 1, i) for i in range(16)}), 16)

    def test_rollout_uses_exact_8_plus_8_chunks_and_restores_mode(self) -> None:
        model = _Model()
        calls = []

        def wrapped(*args, **kwargs):
            self.assertTrue(model.training)
            calls.append(tuple(kwargs["candidate_indices"]))
            return _fake_chunk(*args, **kwargs)

        with patch("ksllm4rec_rloo.rollout._sample_chunk", side_effect=wrapped):
            result = rollout_group(
                model,
                [1, 2],
                _Grammar(),
                group_id="g",
                epoch_index=0,
                chunk_size=8,
                max_completion_length=8,
                device="cpu",
            )
        self.assertEqual(calls, [tuple(range(8)), tuple(range(8, 16))])
        self.assertEqual([row.candidate_index for row in result], list(range(16)))
        self.assertTrue(model.training)

    def test_other_chunk_sizes_are_rejected_by_contract(self) -> None:
        with self.assertRaisesRegex(ValueError, "chunk_size=8"):
            rollout_group(
                _Model(),
                [1],
                _Grammar(),
                group_id="g",
                epoch_index=0,
                chunk_size=4,
                device="cpu",
            )

    def test_padding_exposes_ragged_logps_and_decision_mask(self) -> None:
        rows = tuple(
            RolloutCandidate(
                candidate_index=i,
                sid=Sid("video", 1, 2, 3),
                token_ids=(i + 1,) * (2 if i % 2 else 3),
                old_log_probs=(-0.5,) * (2 if i % 2 else 3),
                decision_mask=(True,) + (False,) * ((2 if i % 2 else 3) - 1),
            )
            for i in range(16)
        )
        batch = pad_rollout_candidates(rows)
        self.assertEqual(batch.old_log_probs.shape, (16, 3))
        self.assertEqual(batch.valid_mask.shape, (16, 3))
        self.assertEqual(batch.decision_mask.shape, (16, 3))
        self.assertTrue(bool((batch.decision_mask <= batch.valid_mask).all()))


if __name__ == "__main__":
    unittest.main()
