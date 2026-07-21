"""Autograd-compatible legal-prefix scoring for the reference-free RLOO run."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .probability import legal_token_log_prob


@dataclass(frozen=True)
class CompletionScores:
    """Ragged completion log-probabilities represented by padded tensors.

    All tensors have shape ``(C, T_max)`` where ``C`` is the number of
    completions in this chunk.  ``valid_mask`` marks real completion tokens;
    ``decision_mask`` is a subset that excludes grammar nodes with exactly one
    legal child.  Padding values in ``log_probs`` are zero and must be ignored
    through the masks.
    """

    log_probs: torch.Tensor
    valid_mask: torch.Tensor
    decision_mask: torch.Tensor

    def __post_init__(self) -> None:
        if self.log_probs.ndim != 2:
            raise ValueError("log_probs must have shape (C, T).")
        if self.valid_mask.shape != self.log_probs.shape:
            raise ValueError("valid_mask must match log_probs shape.")
        if self.decision_mask.shape != self.log_probs.shape:
            raise ValueError("decision_mask must match log_probs shape.")
        if self.valid_mask.dtype != torch.bool or self.decision_mask.dtype != torch.bool:
            raise ValueError("completion masks must be boolean tensors.")
        if bool((self.decision_mask & ~self.valid_mask).any()):
            raise ValueError("decision_mask cannot include padding.")

    @property
    def token_log_probs(self) -> torch.Tensor:
        """Compatibility alias used by objective code."""

        return self.log_probs


def unwrap_causal_lm(model: torch.nn.Module) -> torch.nn.Module:
    """Return the causal LM hidden backbone and language-model head owner."""

    getter = getattr(model, "get_base_model", None)
    causal_lm = getter() if callable(getter) else model
    if not hasattr(causal_lm, "lm_head"):
        raise TypeError(f"Unsupported causal LM wrapper: {type(model)!r}")
    # OneReason uses ``model`` for its transformer backbone.  A few test/fake
    # models expose ``transformer`` instead; accept both without changing the
    # production path.
    if not hasattr(causal_lm, "model") and not hasattr(causal_lm, "transformer"):
        raise TypeError(f"Unsupported causal LM backbone: {type(causal_lm)!r}")
    return causal_lm


def completion_prediction_hidden(
    hidden_states: torch.Tensor,
    *,
    prompt_length: int,
    completion_length: int,
) -> torch.Tensor:
    """Select positions ``P-1:P-1+T`` predicting completion tokens ``0:T``."""

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape (C, L, H).")
    if prompt_length < 1 or completion_length < 1:
        raise ValueError("prompt_length and completion_length must be positive.")
    start = int(prompt_length) - 1
    stop = start + int(completion_length)
    if stop > hidden_states.size(1):
        raise ValueError("Hidden sequence is too short for the requested completion.")
    return hidden_states[:, start:stop, :]


def _backbone(causal_lm: torch.nn.Module) -> torch.nn.Module:
    backbone = getattr(causal_lm, "model", None)
    return backbone if backbone is not None else getattr(causal_lm, "transformer", None)


def _forward_hidden(
    model: torch.nn.Module,
    causal_lm: torch.nn.Module | None = None,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run one full forward and capture hidden states without using a cache."""

    if causal_lm is None:
        causal_lm = unwrap_causal_lm(model)
    backbone = _backbone(causal_lm)
    captured: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, _args: tuple[Any, ...], output: Any) -> None:
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is None and isinstance(output, torch.Tensor):
            hidden = output
        if hidden is None and isinstance(output, (tuple, list)) and output:
            hidden = output[0]
        if hidden is not None:
            captured.append(hidden)

    hook = backbone.register_forward_hook(capture)
    try:
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "use_cache": False,
            "return_dict": True,
            # OneReason accepts this optimization.  The fallback keeps tiny
            # test models and older Transformers usable.
            "logits_to_keep": 1,
        }
        try:
            model(**kwargs)
        except TypeError as exc:
            if "logits_to_keep" not in str(exc):
                raise
            kwargs.pop("logits_to_keep")
            model(**kwargs)
    finally:
        hook.remove()
    if captured:
        return captured[-1]

    # A model may not expose a hookable backbone but return hidden states from
    # its forward.  This branch is intentionally a fallback, not the normal
    # OneReason path.
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
        output_hidden_states=True,
    )
    hidden = getattr(outputs, "last_hidden_state", None)
    if hidden is None:
        states = getattr(outputs, "hidden_states", None)
        hidden = states[-1] if states else None
    if hidden is None:
        raise RuntimeError("Model forward did not expose hidden states.")
    return hidden


def legal_logits_from_hidden(
    causal_lm: torch.nn.Module,
    hidden: torch.Tensor,
    allowed_ids: Sequence[int],
) -> torch.Tensor:
    """Project one hidden vector onto legal vocabulary rows only."""

    if hidden.ndim != 1:
        raise ValueError("hidden must have shape (H,).")
    ids = torch.as_tensor(
        [int(value) for value in allowed_ids], dtype=torch.long, device=hidden.device
    )
    if ids.ndim != 1 or ids.numel() == 0:
        raise ValueError("allowed_ids must be a non-empty one-dimensional sequence.")
    head = causal_lm.lm_head
    weight = head.weight.index_select(0, ids)
    bias = getattr(head, "bias", None)
    legal_bias = bias.index_select(0, ids) if bias is not None else None
    return F.linear(hidden, weight, legal_bias)


def _pad_rows(
    values: list[torch.Tensor], decisions: list[bool], maximum: int, *, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not values:
        raise ValueError("at least one completion row is required")
    width = len(values)
    if width > maximum:
        raise ValueError("completion row exceeds requested maximum width")
    if width:
        row = torch.stack(values)
    else:  # pragma: no cover - guarded by caller
        row = torch.zeros((0,), dtype=torch.float32, device=device)
    row = F.pad(row, (0, maximum - width), value=0.0)
    valid = torch.tensor(
        [True] * width + [False] * (maximum - width), dtype=torch.bool, device=device
    )
    decision = torch.tensor(
        decisions + [False] * (maximum - width), dtype=torch.bool, device=device
    )
    return row, valid, decision


def _score_rows_stepwise(
    model: torch.nn.Module,
    causal_lm: torch.nn.Module,
    prompt: list[int],
    completions: list[list[int]],
    grammar: Any,
    *,
    device: torch.device,
    temperature: float = 1.0,
) -> CompletionScores:
    """Replay rows with the same active-branch layout as the live sampler.

    ``positions[row]`` is the next target index.  Before every model forward,
    each row advances through all singleton grammar nodes (zero logp), leaving
    only branching rows in the padded batch.  This is deliberately stepwise:
    it makes the scorer's prefix, padding, and position layout identical to
    rollout and avoids a second probability definition for replay.
    """

    maximum = max(len(row) for row in completions)
    values: list[list[torch.Tensor | None]] = [
        [None] * len(row) for row in completions
    ]
    decisions: list[list[bool]] = [[] for _ in completions]
    positions = [0] * len(completions)
    prefixes: list[list[int]] = [[] for _ in completions]
    finished = [False] * len(completions)
    eos_id = int(grammar.eos_token_id)

    while not all(finished):
        active: list[int] = []
        # Consume deterministic target tokens without a forward pass.
        for row_index, completion in enumerate(completions):
            if finished[row_index]:
                continue
            while not finished[row_index]:
                target_index = positions[row_index]
                if target_index >= len(completion):
                    raise ValueError(
                        f"Completion row {row_index} ended before grammar EOS."
                    )
                allowed = [
                    int(value) for value in grammar.allowed_next(prefixes[row_index])
                ]
                if not allowed:
                    raise ValueError(
                        f"Completion row {row_index} has tokens after grammar termination."
                    )
                target_id = int(completion[target_index])
                if target_id not in allowed:
                    raise ValueError(
                        f"Completion row {row_index} token {target_index} is illegal."
                    )
                if len(allowed) != 1:
                    active.append(row_index)
                    break
                values[row_index][target_index] = torch.zeros(
                    (), dtype=torch.float32, device=device
                )
                decisions[row_index].append(False)
                prefixes[row_index].append(target_id)
                positions[row_index] += 1
                finished[row_index] = target_id == eos_id
        if not active:
            break

        # Exactly the active rows participate, preserving original row order;
        # right padding and attention masks match _sample_chunk.
        active_prefixes = [prompt + prefixes[index] for index in active]
        width = max(len(row) for row in active_prefixes)
        pad_id = eos_id
        input_ids = torch.tensor(
            [row + [pad_id] * (width - len(row)) for row in active_prefixes],
            dtype=torch.long,
            device=device,
        )
        attention_mask = torch.tensor(
            [[1] * len(row) + [0] * (width - len(row)) for row in active_prefixes],
            dtype=torch.long,
            device=device,
        )
        hidden = _forward_hidden(
            model,
            causal_lm,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        for active_row, row_index in enumerate(active):
            target_index = positions[row_index]
            target_id = int(completions[row_index][target_index])
            allowed = [
                int(value) for value in grammar.allowed_next(prefixes[row_index])
            ]
            if len(allowed) <= 1 or target_id not in allowed:
                raise RuntimeError("Replay active row is no longer branching/legal.")
            predecessor = len(active_prefixes[active_row]) - 1
            legal_logits = legal_logits_from_hidden(
                causal_lm, hidden[active_row, predecessor], allowed
            )
            values[row_index][target_index] = legal_token_log_prob(
                legal_logits,
                list(range(len(allowed))),
                allowed.index(target_id),
                temperature=temperature,
            )
            decisions[row_index].append(True)
            prefixes[row_index].append(target_id)
            positions[row_index] += 1
            finished[row_index] = target_id == eos_id

    log_rows: list[torch.Tensor] = []
    valid_rows: list[torch.Tensor] = []
    decision_rows: list[torch.Tensor] = []
    for row_index, completion in enumerate(completions):
        if not finished[row_index] or positions[row_index] != len(completion):
            raise ValueError(f"Completion row {row_index} failed grammar termination.")
        grammar.parse(prefixes[row_index])
        # ``values`` is fully populated at real token positions.  The explicit
        # assertion catches an accidental missing branch before it reaches an
        # objective reduction.
        if any(value is None for value in values[row_index]):
            raise RuntimeError("Replay left an unscored completion token.")
        row_values = [value for value in values[row_index] if value is not None]
        padded, valid, decision = _pad_rows(
            row_values, decisions[row_index], maximum, device=device
        )
        log_rows.append(padded)
        valid_rows.append(valid)
        decision_rows.append(decision)
    return CompletionScores(
        log_probs=torch.stack(log_rows),
        valid_mask=torch.stack(valid_rows),
        decision_mask=torch.stack(decision_rows),
    )


def score_completions(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[Sequence[int]],
    grammar: Any,
    *,
    device: torch.device | str,
    temperature: float = 1.0,
) -> CompletionScores:
    """Score a ragged completion batch under the exact legal denominators."""

    prompt = [int(value) for value in prompt_ids]
    completions = [[int(value) for value in row] for row in completion_ids]
    if not prompt:
        raise ValueError("prompt_ids must be non-empty.")
    if not completions or any(not row for row in completions):
        raise ValueError("completion_ids must contain non-empty rows.")
    torch_device = torch.device(device)
    causal_lm = unwrap_causal_lm(model)
    return _score_rows_stepwise(
        model,
        causal_lm,
        prompt,
        completions,
        grammar,
        device=torch_device,
        temperature=temperature,
    )


def score_completions_chunked(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[Sequence[int]],
    grammar: Any,
    *,
    chunk_size: int = 8,
    device: torch.device | str,
    temperature: float = 1.0,
) -> tuple[CompletionScores, ...]:
    """Score completion chunks independently, without aggregating GT mass.

    The caller (objective) must concatenate sequence log-probabilities from all
    returned chunks before applying a single global ``logsumexp`` for a GT set.
    """

    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if len(completion_ids) == 16 and int(chunk_size) != 8:
        raise ValueError("G=16 completion scoring requires chunks of 8.")
    rows = [[int(value) for value in row] for row in completion_ids]
    if not rows:
        raise ValueError("completion_ids must not be empty.")
    result: list[CompletionScores] = []
    for start in range(0, len(rows), int(chunk_size)):
        result.append(
            score_completions(
                model,
                prompt_ids,
                rows[start : start + int(chunk_size)],
                grammar,
                device=device,
                temperature=temperature,
            )
        )
    return tuple(result)


# Semantic alias for the anchor path; it deliberately returns chunks rather
# than a pre-aggregated loss.
score_gt_completions = score_completions_chunked


__all__ = [
    "CompletionScores",
    "_forward_hidden",
    "completion_prediction_hidden",
    "legal_logits_from_hidden",
    "score_completions",
    "score_completions_chunked",
    "score_gt_completions",
    "unwrap_causal_lm",
]
