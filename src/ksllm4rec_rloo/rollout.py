"""G=16 live constrained rollout in two memory-bounded chunks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence

import torch

from ksllm4rec_orpo.data import Sid

from .probability import sample_legal_action
from .scoring import _forward_hidden, legal_logits_from_hidden, unwrap_causal_lm


NUM_CANDIDATES = 16
CHUNK_SIZE = 8


@dataclass(frozen=True)
class RolloutCandidate:
    """One legal live completion and its detached sampling-time statistics."""

    candidate_index: int
    sid: Sid
    token_ids: tuple[int, ...]
    old_log_probs: tuple[float, ...]
    decision_mask: tuple[bool, ...]

    @property
    def completion_ids(self) -> tuple[int, ...]:
        return self.token_ids

    @property
    def log_probs(self) -> tuple[float, ...]:
        """Alias for callers that use the shorter replay-statistics name."""

        return self.old_log_probs

    @property
    def old_logp(self) -> tuple[float, ...]:
        return self.old_log_probs


@dataclass(frozen=True)
class RolloutBatch:
    """Ragged candidates plus padded log-probability/mask tensors."""

    candidates: tuple[RolloutCandidate, ...]
    token_ids: torch.Tensor
    old_log_probs: torch.Tensor
    valid_mask: torch.Tensor
    decision_mask: torch.Tensor

    @property
    def log_probs(self) -> torch.Tensor:
        return self.old_log_probs

    @property
    def padded_token_ids(self) -> torch.Tensor:
        return self.token_ids


def rollout_seed(group_id: str, epoch_index: int, candidate_index: int) -> int:
    """Return the frozen unsigned-64 SHA256 seed for one candidate."""

    if not isinstance(group_id, str) or not group_id:
        raise ValueError("group_id must be a non-empty string.")
    if isinstance(epoch_index, bool) or int(epoch_index) < 0:
        raise ValueError("epoch_index must be a non-negative integer.")
    if isinstance(candidate_index, bool) or not 0 <= int(candidate_index) < NUM_CANDIDATES:
        raise ValueError("candidate_index must be in 0..15.")
    payload = f"rollout|42|{int(epoch_index)}|{group_id}|{int(candidate_index)}".encode(
        "utf-8"
    )
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big", signed=False)


def _device_generator(device: torch.device, seed: int) -> torch.Generator:
    try:
        generator = torch.Generator(device=device)
    except (RuntimeError, TypeError):
        # CPU tests can pass a device alias unsupported by the local torch
        # build; this fallback does not affect CUDA production runs.
        generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
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
    temperature: float = 1.0,
) -> list[RolloutCandidate]:
    """Sample one chunk while retaining the exact per-token old log-probs."""

    indices = [int(value) for value in candidate_indices]
    if not indices or len(indices) > CHUNK_SIZE:
        raise ValueError("candidate_indices must contain one to eight entries.")
    if indices != list(range(indices[0], indices[0] + len(indices))):
        raise ValueError("candidate_indices must be contiguous.")
    if any(index < 0 or index >= NUM_CANDIDATES for index in indices):
        raise ValueError("candidate_indices must be in 0..15.")
    if max_completion_length < 1:
        raise ValueError("max_completion_length must be positive.")
    prompt = [int(value) for value in prompt_ids]
    if not prompt:
        raise ValueError("prompt_ids must not be empty.")

    causal_lm = unwrap_causal_lm(model)
    generators = {
        index: _device_generator(device, rollout_seed(group_id, epoch_index, index))
        for index in indices
    }
    tokens: dict[int, list[int]] = {index: [] for index in indices}
    logps: dict[int, list[float]] = {index: [] for index in indices}
    decisions: dict[int, list[bool]] = {index: [] for index in indices}
    finished = {index: False for index in indices}
    eos_id = int(grammar.eos_token_id)

    while not all(finished.values()):
        # Advance deterministic grammar chains before touching the model.  A
        # singleton node has probability one, so this both saves a forward and
        # guarantees that it consumes no RNG state.
        active: list[int] = []
        for index in indices:
            if finished[index]:
                continue
            while not finished[index]:
                if len(tokens[index]) >= int(max_completion_length):
                    raise RuntimeError(
                        "Constrained rollout exceeded "
                        f"max_completion_length={max_completion_length}."
                    )
                allowed = [
                    int(value) for value in grammar.allowed_next(tokens[index])
                ]
                if not allowed:
                    raise RuntimeError("Grammar ended without emitting EOS.")
                if len(allowed) != 1:
                    active.append(index)
                    break
                token_id = allowed[0]
                tokens[index].append(token_id)
                logps[index].append(0.0)
                decisions[index].append(False)
                finished[index] = token_id == eos_id
        if not active:
            break

        # Active rows retain candidate order.  The scorer uses the same active
        # row filtering and right-padding layout when replaying these prefixes.
        prefixes = [prompt + tokens[index] for index in active]
        maximum = max(len(row) for row in prefixes)
        input_ids = torch.tensor(
            [row + [eos_id] * (maximum - len(row)) for row in prefixes],
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
        for row_index, index in enumerate(active):
            generated = tokens[index]
            allowed = [int(value) for value in grammar.allowed_next(generated)]
            if len(allowed) <= 1:
                raise RuntimeError("Active rollout row lost its branching node.")
            predecessor = len(prefixes[row_index]) - 1
            legal_logits = legal_logits_from_hidden(
                causal_lm, hidden[row_index, predecessor], allowed
            )
            # Use local IDs because legal_logits is already projected in the
            # exact order returned by grammar.allowed_next().
            local_token, log_prob, decision = sample_legal_action(
                legal_logits,
                list(range(len(allowed))),
                generator=generators[index],
                temperature=temperature,
            )
            token_id = int(allowed[local_token])
            generated.append(token_id)
            logps[index].append(float(log_prob.detach().float().item()))
            decisions[index].append(bool(decision))
            finished[index] = token_id == eos_id
    if not all(finished.values()):  # pragma: no cover - guarded by loop
        raise RuntimeError("Constrained rollout ended before all rows emitted EOS.")

    result: list[RolloutCandidate] = []
    for index in indices:
        row_tokens = tokens[index]
        if not row_tokens or row_tokens[-1] != eos_id:
            raise RuntimeError("A rollout did not terminate with EOS.")
        sid = grammar.parse(row_tokens)
        result.append(
            RolloutCandidate(
                candidate_index=index,
                sid=sid,
                token_ids=tuple(row_tokens),
                old_log_probs=tuple(logps[index]),
                decision_mask=tuple(decisions[index]),
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
    chunk_size: int = CHUNK_SIZE,
    num_candidates: int = NUM_CANDIDATES,
    max_completion_length: int = 32,
    device: torch.device | str | None = None,
    temperature: float = 1.0,
) -> tuple[RolloutCandidate, ...]:
    """Generate exactly candidates 0..15 using fixed chunks 0..7 and 8..15."""

    if int(num_candidates) != NUM_CANDIDATES:
        raise ValueError("Spec V2.0 requires exactly 16 candidates.")
    if int(chunk_size) != CHUNK_SIZE:
        raise ValueError("Spec V2.0 requires chunk_size=8 (the 8+8 layout).")
    if device is None:
        parameter = next(model.parameters(), None)
        torch_device = parameter.device if parameter is not None else torch.device("cpu")
    else:
        torch_device = torch.device(device)
    # Keep the caller-selected mode.  Training uses model.train() with every
    # dropout source zeroed, so no-grad rollout and gradient replay share one
    # kernel path while replay still activates gradient checkpointing.
    candidates: list[RolloutCandidate] = []
    for start in (0, CHUNK_SIZE):
        candidates.extend(
            _sample_chunk(
                model,
                prompt_ids,
                grammar,
                group_id=group_id,
                epoch_index=epoch_index,
                candidate_indices=list(range(start, start + CHUNK_SIZE)),
                max_completion_length=max_completion_length,
                device=torch_device,
                temperature=temperature,
            )
        )
    candidates.sort(key=lambda candidate: candidate.candidate_index)
    if [candidate.candidate_index for candidate in candidates] != list(
        range(NUM_CANDIDATES)
    ):
        raise RuntimeError("Rollout did not return exactly candidate indices 0..15.")
    return tuple(candidates)


def pad_rollout_candidates(
    candidates: Sequence[RolloutCandidate],
    *,
    device: torch.device | str | None = None,
    pad_token_id: int = 0,
) -> RolloutBatch:
    """Convert ragged rollout rows to tensors suitable for RLOO loss."""

    rows = tuple(candidates)
    if len(rows) != NUM_CANDIDATES:
        raise ValueError("Exactly 16 rollout candidates are required.")
    indices = [row.candidate_index for row in rows]
    if indices != list(range(NUM_CANDIDATES)):
        raise ValueError("Candidates must be ordered by index 0..15.")
    if any(not row.token_ids for row in rows):
        raise ValueError("Every rollout candidate must contain a completion.")
    if any(len(row.token_ids) != len(row.old_log_probs) for row in rows):
        raise ValueError("token_ids and old_log_probs lengths must match.")
    if any(len(row.token_ids) != len(row.decision_mask) for row in rows):
        raise ValueError("token_ids and decision_mask lengths must match.")
    maximum = max(len(row.token_ids) for row in rows)
    target_device = torch.device(device) if device is not None else torch.device("cpu")
    token_tensor = torch.full(
        (NUM_CANDIDATES, maximum), int(pad_token_id), dtype=torch.long, device=target_device
    )
    log_tensor = torch.zeros(
        (NUM_CANDIDATES, maximum), dtype=torch.float32, device=target_device
    )
    valid_tensor = torch.zeros(
        (NUM_CANDIDATES, maximum), dtype=torch.bool, device=target_device
    )
    decision_tensor = torch.zeros(
        (NUM_CANDIDATES, maximum), dtype=torch.bool, device=target_device
    )
    for row_index, row in enumerate(rows):
        length = len(row.token_ids)
        token_tensor[row_index, :length] = torch.as_tensor(
            row.token_ids, dtype=torch.long, device=target_device
        )
        log_tensor[row_index, :length] = torch.as_tensor(
            row.old_log_probs, dtype=torch.float32, device=target_device
        )
        valid_tensor[row_index, :length] = True
        decision_tensor[row_index, :length] = torch.as_tensor(
            row.decision_mask, dtype=torch.bool, device=target_device
        )
    return RolloutBatch(
        candidates=rows,
        token_ids=token_tensor,
        old_log_probs=log_tensor,
        valid_mask=valid_tensor,
        decision_mask=decision_tensor,
    )


def rollout_group_padded(
    *args: Any,
    pad_device: torch.device | str | None = None,
    pad_token_id: int = 0,
    **kwargs: Any,
) -> RolloutBatch:
    """Run :func:`rollout_group` and return its padded tensor representation."""

    if pad_device is None and "device" in kwargs:
        pad_device = kwargs["device"]
    return pad_rollout_candidates(
        rollout_group(*args, **kwargs),
        device=pad_device,
        pad_token_id=pad_token_id,
    )


__all__ = [
    "CHUNK_SIZE",
    "NUM_CANDIDATES",
    "RolloutBatch",
    "RolloutCandidate",
    "pad_rollout_candidates",
    "rollout_group",
    "rollout_group_padded",
    "rollout_seed",
]
