"""Fail-closed constrained SID Pass@64 evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from transformers import LogitsProcessorList

from ._infra.grpo_constraint import RecommendationLogitsProcessor
from ._infra.orpo_data import Sid


_TASK_GRAMMAR_MODES = {
    "recommend": "probe_recommendation",
    "recommendation": "probe_recommendation",
    "text_to_sid": "probe_text_to_sid",
    "item_text_to_sid": "probe_text_to_sid",
}


class Probe64ContractError(RuntimeError):
    """The evaluator did not receive one complete legal 64-candidate set."""


@dataclass(frozen=True)
class Probe64Result:
    """All 64 consumed SID candidates and the first one matching the target."""

    task: str
    target_sid: str
    candidate_sids: tuple[str, ...]
    hit_rank: int | None
    pass_at_64: bool


def run_probe64(
    model: Any,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    grammar: Any,
    task: str,
    target_sid: Sid,
    max_new_tokens: int,
    num_beams: int = 64,
    num_return_sequences: int = 64,
    cache_implementation: str = "offloaded",
) -> Probe64Result:
    """Generate, parse, and consume exactly 64 constrained beam candidates.

    ``input_ids`` has shape ``(1, prompt_tokens)``.  ``model.generate`` must
    return shape ``(64, prompt_tokens + completion_tokens)``.
    """

    task_name = str(task)
    expected_mode = _TASK_GRAMMAR_MODES.get(task_name)
    if expected_mode is None:
        raise Probe64ContractError(f"Unsupported probe task: {task_name!r}.")
    if getattr(grammar, "mode", None) != expected_mode:
        raise Probe64ContractError(
            f"Task {task_name!r} requires grammar mode {expected_mode!r}."
        )
    if int(num_beams) != 64:
        raise Probe64ContractError("Pass@64 requires num_beams=64.")
    if int(num_return_sequences) != 64:
        raise Probe64ContractError(
            "Pass@64 requires num_return_sequences=64."
        )
    if cache_implementation != "offloaded":
        raise Probe64ContractError(
            "Pass@64 requires the exact BF16 offloaded KV cache on 24 GiB GPUs."
        )
    prompt_length = int(input_ids.shape[1])
    processor = RecommendationLogitsProcessor(grammar, prompt_length)
    generated = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=int(max_new_tokens),
        do_sample=False,
        num_beams=int(num_beams),
        num_return_sequences=int(num_return_sequences),
        renormalize_logits=True,
        early_stopping=True,
        use_cache=True,
        cache_implementation=cache_implementation,
        pad_token_id=grammar.eos_token_id,
        eos_token_id=grammar.eos_token_id,
        logits_processor=LogitsProcessorList([processor]),
    )
    if len(generated) != 64:
        raise Probe64ContractError(
            f"Pass@64 requires exactly 64 returned sequences, got {len(generated)}."
        )
    parsed: list[str] = []
    for rank, row in enumerate(generated, start=1):
        try:
            parsed.append(grammar.parse(row[prompt_length:].tolist()).render())
        except Exception as exc:
            raise Probe64ContractError(
                f"Pass@64 received illegal candidate {rank}."
            ) from exc
    candidates = tuple(parsed)
    if len(set(candidates)) != 64:
        raise Probe64ContractError(
            "Pass@64 requires 64 different SIDs; duplicate candidates are invalid."
        )
    target = target_sid.render()
    hit_rank = next(
        (
            index
            for index, candidate in enumerate(candidates, start=1)
            if candidate == target
        ),
        None,
    )
    return Probe64Result(
        task=task_name,
        target_sid=target,
        candidate_sids=candidates,
        hit_rank=hit_rank,
        pass_at_64=hit_rank is not None,
    )


__all__ = ["Probe64ContractError", "Probe64Result", "run_probe64"]
