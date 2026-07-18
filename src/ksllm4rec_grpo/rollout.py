"""On-policy constrained sampling for exactly eight GRPO candidates."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from ksllm4rec_orpo.data import Sid

from .probability import sample_legal_token
from .scoring import (
    _forward_hidden,
    legal_logits_from_hidden,
    unwrap_causal_lm,
)


@dataclass(frozen=True)
class RolloutCandidate:
    candidate_index: int
    sid: Sid
    token_ids: tuple[int, ...]
    old_log_probs: tuple[float, ...]
    decision_mask: tuple[bool, ...]


def rollout_seed(group_id: str, epoch_index: int, candidate_index: int) -> int:
    """Map the frozen SHA256 payload to one independent 64-bit generator seed."""

    if not group_id or epoch_index <= 0 or candidate_index < 0:
        raise ValueError("Invalid rollout seed coordinates.")
    payload = f"rollout|42|{epoch_index}|{group_id}|{candidate_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _device_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return generator


@torch.no_grad()
def _sample_chunk(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    grammar: Any,
    *,
    group_id: str,
    epoch_index: int,
    candidate_indices: Sequence[int],
    max_completion_length: int,
    device: torch.device,
) -> list[RolloutCandidate]:
    count = len(candidate_indices)
    if count < 1:
        return []
    prompt = [int(value) for value in prompt_ids]
    causal_lm = unwrap_causal_lm(model)
    generators = [
        _device_generator(device, rollout_seed(group_id, epoch_index, candidate_index))
        for candidate_index in candidate_indices
    ]
    tokens: list[list[int]] = [[] for _ in range(count)]
    logps: list[list[float]] = [[] for _ in range(count)]
    decisions: list[list[bool]] = [[] for _ in range(count)]
    finished = [False] * count

    for step in range(max_completion_length):
        prefixes = [prompt + row for row in tokens]
        maximum = max(len(row) for row in prefixes)
        pad_id = int(grammar.eos_token_id)
        input_ids = torch.tensor(
            [row + [pad_id] * (maximum - len(row)) for row in prefixes],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.tensor(
            [[1] * len(row) + [0] * (maximum - len(row)) for row in prefixes],
            dtype=torch.long,
            device=device,
        )
        hidden = _forward_hidden(
            model,
            causal_lm,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        for row in range(count):
            if finished[row]:
                continue
            allowed = grammar.allowed_next(tokens[row])
            if not allowed:
                raise RuntimeError("Grammar ended without emitting EOS.")
            predecessor = len(prefixes[row]) - 1
            legal_logits = legal_logits_from_hidden(
                causal_lm, hidden[row, predecessor], allowed
            )
            local_token, logp = sample_legal_token(
                legal_logits,
                list(range(len(allowed))),
                generator=generators[row],
            )
            token_id = int(allowed[local_token])
            tokens[row].append(token_id)
            logps[row].append(float(logp.item()))
            decisions[row].append(len(allowed) > 1)
            finished[row] = token_id == int(grammar.eos_token_id)
        if all(finished):
            break
    if not all(finished):
        raise RuntimeError(
            f"Constrained rollout exceeded max_completion_length={max_completion_length}."
        )

    result = []
    for row, candidate_index in enumerate(candidate_indices):
        sid = grammar.parse(tokens[row])
        result.append(
            RolloutCandidate(
                candidate_index=int(candidate_index),
                sid=sid,
                token_ids=tuple(tokens[row]),
                old_log_probs=tuple(logps[row]),
                decision_mask=tuple(decisions[row]),
            )
        )
    return result


def rollout_group(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    grammar: Any,
    *,
    group_id: str,
    epoch_index: int,
    chunk_size: int,
    max_completion_length: int,
    device: torch.device | str,
) -> tuple[RolloutCandidate, ...]:
    """Live-sample candidates 0..7; duplicates are retained by design."""

    if chunk_size not in (1, 2, 4, 8):
        raise ValueError("chunk_size must be one of 8, 4, 2, 1.")
    torch_device = torch.device(device)
    was_training = model.training
    model.eval()
    try:
        candidates: list[RolloutCandidate] = []
        for start in range(0, 8, chunk_size):
            indices = list(range(start, min(start + chunk_size, 8)))
            candidates.extend(
                _sample_chunk(
                    model,
                    prompt_ids,
                    grammar,
                    group_id=group_id,
                    epoch_index=epoch_index,
                    candidate_indices=indices,
                    max_completion_length=max_completion_length,
                    device=torch_device,
                )
            )
    finally:
        model.train(was_training)
    if [candidate.candidate_index for candidate in candidates] != list(range(8)):
        raise RuntimeError("Rollout did not return exactly candidate indices 0..7.")
    return tuple(candidates)
