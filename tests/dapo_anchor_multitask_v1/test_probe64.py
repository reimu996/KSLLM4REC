from __future__ import annotations

import unittest

import torch

from ksllm4rec_dapo_anchor_multitask_v1._infra.orpo_data import Sid
from ksllm4rec_dapo_anchor_multitask_v1.probe64 import (
    Probe64ContractError,
    run_probe64,
)


class _FakeGrammar:
    def __init__(self, *, mode: str) -> None:
        self.mode = mode
        self.eos_token_id = 0
        self.parsed: list[tuple[int, ...]] = []

    def parse(self, completion: list[int]) -> Sid:
        value = tuple(int(token) for token in completion)
        self.parsed.append(value)
        return Sid("video", 0, 0, value[0])


class _InvalidLastGrammar(_FakeGrammar):
    def parse(self, completion: list[int]) -> Sid:
        if completion == [63]:
            raise ValueError("not in trie")
        return super().parse(completion)


class _FakeModel:
    def __init__(self, generated: torch.Tensor) -> None:
        self.generated = generated
        self.kwargs: dict | None = None

    def generate(self, **kwargs):
        self.kwargs = kwargs
        return self.generated


def _generated(count: int = 64) -> torch.Tensor:
    return torch.tensor([[91, index] for index in range(count)], dtype=torch.long)


class Probe64Test(unittest.TestCase):
    def test_consumes_all_64_unique_candidates_and_finds_gt_at_rank_37(self) -> None:
        model = _FakeModel(_generated())
        grammar = _FakeGrammar(mode="probe_recommendation")

        result = run_probe64(
            model,
            input_ids=torch.tensor([[91]], dtype=torch.long),
            attention_mask=torch.tensor([[1]], dtype=torch.long),
            grammar=grammar,
            task="recommendation",
            target_sid=Sid("video", 0, 0, 36),
            max_new_tokens=8,
            num_beams=64,
            num_return_sequences=64,
        )

        self.assertEqual(model.kwargs["num_beams"], 64)
        self.assertEqual(model.kwargs["num_return_sequences"], 64)
        self.assertEqual(model.kwargs["cache_implementation"], "offloaded")
        self.assertEqual(len(grammar.parsed), 64)
        self.assertEqual(len(result.candidate_sids), 64)
        self.assertEqual(result.hit_rank, 37)
        self.assertTrue(result.pass_at_64)

    def test_63_returned_sequences_fail_closed(self) -> None:
        with self.assertRaisesRegex(Probe64ContractError, "exactly 64"):
            run_probe64(
                _FakeModel(_generated(63)),
                input_ids=torch.tensor([[91]], dtype=torch.long),
                attention_mask=torch.tensor([[1]], dtype=torch.long),
                grammar=_FakeGrammar(mode="probe_recommendation"),
                task="recommendation",
                target_sid=Sid("video", 0, 0, 36),
                max_new_tokens=8,
                num_beams=64,
                num_return_sequences=64,
            )

    def test_duplicate_sid_fails_closed(self) -> None:
        generated = _generated()
        generated[-1, -1] = generated[0, -1]
        with self.assertRaisesRegex(Probe64ContractError, "different SIDs"):
            run_probe64(
                _FakeModel(generated),
                input_ids=torch.tensor([[91]], dtype=torch.long),
                attention_mask=torch.tensor([[1]], dtype=torch.long),
                grammar=_FakeGrammar(mode="probe_recommendation"),
                task="recommendation",
                target_sid=Sid("video", 0, 0, 36),
                max_new_tokens=8,
                num_beams=64,
                num_return_sequences=64,
            )

    def test_one_illegal_sid_fails_closed(self) -> None:
        with self.assertRaisesRegex(Probe64ContractError, "illegal candidate 64"):
            run_probe64(
                _FakeModel(_generated()),
                input_ids=torch.tensor([[91]], dtype=torch.long),
                attention_mask=torch.tensor([[1]], dtype=torch.long),
                grammar=_InvalidLastGrammar(mode="probe_recommendation"),
                task="recommendation",
                target_sid=Sid("video", 0, 0, 36),
                max_new_tokens=8,
                num_beams=64,
                num_return_sequences=64,
            )

    def test_task_must_use_its_matching_probe_grammar(self) -> None:
        with self.assertRaisesRegex(Probe64ContractError, "probe_text_to_sid"):
            run_probe64(
                _FakeModel(_generated()),
                input_ids=torch.tensor([[91]], dtype=torch.long),
                attention_mask=torch.tensor([[1]], dtype=torch.long),
                grammar=_FakeGrammar(mode="probe_recommendation"),
                task="item_text_to_sid",
                target_sid=Sid("video", 0, 0, 36),
                max_new_tokens=8,
                num_beams=64,
                num_return_sequences=64,
            )

    def test_beam_and_return_counts_are_frozen_at_64(self) -> None:
        with self.assertRaisesRegex(Probe64ContractError, "num_beams=64"):
            run_probe64(
                _FakeModel(_generated()),
                input_ids=torch.tensor([[91]], dtype=torch.long),
                attention_mask=torch.tensor([[1]], dtype=torch.long),
                grammar=_FakeGrammar(mode="probe_text_to_sid"),
                task="text_to_sid",
                target_sid=Sid("video", 0, 0, 36),
                max_new_tokens=8,
                num_beams=32,
                num_return_sequences=64,
            )

        with self.assertRaisesRegex(Probe64ContractError, "num_return_sequences=64"):
            run_probe64(
                _FakeModel(_generated()),
                input_ids=torch.tensor([[91]], dtype=torch.long),
                attention_mask=torch.tensor([[1]], dtype=torch.long),
                grammar=_FakeGrammar(mode="probe_text_to_sid"),
                task="text_to_sid",
                target_sid=Sid("video", 0, 0, 36),
                max_new_tokens=8,
                num_beams=64,
                num_return_sequences=63,
            )

    def test_text_to_sid_probe_grammar_is_supported(self) -> None:
        result = run_probe64(
            _FakeModel(_generated()),
            input_ids=torch.tensor([[91]], dtype=torch.long),
            attention_mask=torch.tensor([[1]], dtype=torch.long),
            grammar=_FakeGrammar(mode="probe_text_to_sid"),
            task="text_to_sid",
            target_sid=Sid("video", 0, 0, 63),
            max_new_tokens=8,
            num_beams=64,
            num_return_sequences=64,
        )

        self.assertEqual(result.task, "text_to_sid")
        self.assertEqual(result.hit_rank, 64)
        self.assertTrue(result.pass_at_64)

    def test_quantized_or_split_cache_modes_are_rejected(self) -> None:
        with self.assertRaisesRegex(Probe64ContractError, "offloaded KV cache"):
            run_probe64(
                _FakeModel(_generated()),
                input_ids=torch.tensor([[91]], dtype=torch.long),
                attention_mask=torch.tensor([[1]], dtype=torch.long),
                grammar=_FakeGrammar(mode="probe_text_to_sid"),
                task="text_to_sid",
                target_sid=Sid("video", 0, 0, 63),
                max_new_tokens=8,
                cache_implementation="quantized",
            )


if __name__ == "__main__":
    unittest.main()
