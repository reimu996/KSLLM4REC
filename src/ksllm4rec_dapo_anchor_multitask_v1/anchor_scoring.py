"""Dense no-cache scoring for heterogeneous prompt/GT completion pairs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from ._infra.dapo_scoring import _forward_hidden_no_cache
from ._infra.rloo_probability import legal_log_probs
from ._infra.rloo_scoring import legal_logits_from_hidden, unwrap_causal_lm


@dataclass(frozen=True)
class HeterogeneousCompletionScores:
    log_probs: torch.Tensor
    valid_mask: torch.Tensor
    decision_mask: torch.Tensor


def _validate_completion(grammar: Any, completion: Sequence[int]) -> tuple[int, ...]:
    values = tuple(int(value) for value in completion)
    if not values:
        raise ValueError("GT completion must not be empty.")
    prefix: list[int] = []
    for index, token in enumerate(values):
        allowed = tuple(int(value) for value in grammar.allowed_next(prefix))
        if token not in allowed:
            raise ValueError(f"GT completion token {index} is illegal.")
        prefix.append(token)
        if token == int(grammar.eos_token_id) and index != len(values) - 1:
            raise ValueError("GT completion contains tokens after EOS.")
    if values[-1] != int(grammar.eos_token_id):
        raise ValueError("GT completion must terminate with EOS.")
    grammar.parse(list(values))
    return values


def score_prompt_completion_pairs_dense(
    model: torch.nn.Module,
    prompt_ids_rows: Sequence[Sequence[int]],
    completion_ids_rows: Sequence[Sequence[int]],
    grammar: Any,
    *,
    device: str | torch.device,
    temperature: float,
    completion_width: int,
    max_batch_rows: int = 8,
) -> HeterogeneousCompletionScores:
    """Score up to eight variable-prompt GT rows in one no-cache forward."""

    prompts = [tuple(int(value) for value in row) for row in prompt_ids_rows]
    completions = [
        _validate_completion(grammar, row) for row in completion_ids_rows
    ]
    if len(prompts) != len(completions) or not 1 <= len(prompts) <= int(max_batch_rows):
        raise ValueError("Prompt/GT rows must have the same size in 1..max_batch_rows.")
    if any(not prompt for prompt in prompts):
        raise ValueError("Every prompt row must be non-empty.")
    if any(len(completion) > int(completion_width) for completion in completions):
        raise ValueError("GT completion exceeds completion_width.")

    torch_device = torch.device(device)
    eos = int(grammar.eos_token_id)
    max_prompt = max(len(prompt) for prompt in prompts)
    total_width = max_prompt + int(completion_width)
    input_rows: list[list[int]] = []
    mask_rows: list[list[int]] = []
    for prompt, completion in zip(prompts, completions, strict=True):
        actual = list(prompt) + list(completion)
        padding = total_width - len(actual)
        input_rows.append(actual + [eos] * padding)
        mask_rows.append([1] * len(actual) + [0] * padding)
    input_ids = torch.tensor(input_rows, dtype=torch.long, device=torch_device)
    attention_mask = torch.tensor(mask_rows, dtype=torch.long, device=torch_device)
    causal_lm = unwrap_causal_lm(model)
    hidden = _forward_hidden_no_cache(
        causal_lm,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )

    score_rows: list[torch.Tensor] = []
    valid_rows: list[torch.Tensor] = []
    decision_rows: list[torch.Tensor] = []
    for row_index, (prompt, completion) in enumerate(
        zip(prompts, completions, strict=True)
    ):
        prefix: list[int] = []
        values: list[torch.Tensor] = []
        decisions: list[bool] = []
        prediction_start = len(prompt) - 1
        for position, token in enumerate(completion):
            allowed = tuple(int(value) for value in grammar.allowed_next(prefix))
            if len(allowed) == 1:
                value = torch.zeros((), dtype=torch.float32, device=torch_device)
                decision = False
            else:
                logits = legal_logits_from_hidden(
                    causal_lm,
                    hidden[row_index, prediction_start + position],
                    allowed,
                )
                probabilities = legal_log_probs(
                    logits,
                    list(range(len(allowed))),
                    temperature=float(temperature),
                )
                value = probabilities[allowed.index(int(token))]
                decision = True
            values.append(value)
            decisions.append(decision)
            prefix.append(int(token))
        padding = int(completion_width) - len(completion)
        score_rows.append(F.pad(torch.stack(values), (0, padding), value=0.0))
        valid_rows.append(
            torch.tensor(
                [True] * len(completion) + [False] * padding,
                dtype=torch.bool,
                device=torch_device,
            )
        )
        decision_rows.append(
            torch.tensor(
                decisions + [False] * padding,
                dtype=torch.bool,
                device=torch_device,
            )
        )
    return HeterogeneousCompletionScores(
        log_probs=torch.stack(score_rows),
        valid_mask=torch.stack(valid_rows),
        decision_mask=torch.stack(decision_rows),
    )


__all__ = ["HeterogeneousCompletionScores", "score_prompt_completion_pairs_dense"]
