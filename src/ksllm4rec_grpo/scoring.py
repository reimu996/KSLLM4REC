"""Memory-bounded, exactly aligned legal-token full-sequence scoring."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
import torch.nn.functional as F

from .probability import legal_token_log_prob


@dataclass(frozen=True)
class CompletionScores:
    """Candidate log-probabilities padded to shape ``(C, T_max)``."""

    log_probs: torch.Tensor
    valid_mask: torch.Tensor
    decision_mask: torch.Tensor


def unwrap_causal_lm(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying Transformers causal LM from a PEFT wrapper."""

    causal_lm = model.get_base_model() if hasattr(model, "get_base_model") else model
    if not hasattr(causal_lm, "model") or not hasattr(causal_lm, "lm_head"):
        raise TypeError(f"Unsupported causal LM wrapper: {type(model)!r}")
    return causal_lm


def completion_prediction_hidden(
    hidden_states: torch.Tensor,
    *,
    prompt_length: int,
    completion_length: int,
) -> torch.Tensor:
    """Select hidden ``P-1:P+T-1`` that predicts completion tokens ``0:T``."""

    if hidden_states.ndim != 3:
        raise ValueError("hidden_states must have shape (C, L, H).")
    if prompt_length < 1 or completion_length < 1:
        raise ValueError("prompt_length and completion_length must be positive.")
    start = prompt_length - 1
    stop = start + completion_length
    if stop > hidden_states.size(1):
        raise ValueError("Hidden sequence is too short for the requested completion.")
    return hidden_states[:, start:stop, :]


def _forward_hidden(
    model: torch.nn.Module,
    causal_lm: torch.nn.Module,
    *,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    """Run a full no-cache forward while retaining backbone hidden states."""

    captured: list[torch.Tensor] = []

    def capture(_module, _args, output) -> None:
        captured.append(output.last_hidden_state)

    hook = causal_lm.model.register_forward_hook(capture)
    try:
        model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            return_dict=True,
            logits_to_keep=1,
        )
    finally:
        hook.remove()
    if len(captured) != 1:
        raise RuntimeError(f"Expected one hidden-state capture, got {len(captured)}.")
    return captured[0]


def legal_logits_from_hidden(
    causal_lm: torch.nn.Module,
    hidden: torch.Tensor,
    allowed_ids: Sequence[int],
) -> torch.Tensor:
    """Project one hidden vector only onto the exact legal token rows."""

    ids = torch.tensor(allowed_ids, dtype=torch.long, device=hidden.device)
    weight = causal_lm.lm_head.weight.index_select(0, ids)
    bias = getattr(causal_lm.lm_head, "bias", None)
    legal_bias = bias.index_select(0, ids) if bias is not None else None
    return F.linear(hidden, weight, legal_bias)


def score_completions(
    model: torch.nn.Module,
    prompt_ids: Sequence[int],
    completion_ids: Sequence[Sequence[int]],
    grammar: Any,
    *,
    device: torch.device | str,
) -> CompletionScores:
    """Teacher-force candidates under each step's exact legal denominator.

    ``prompt_ids`` has shape ``(P,)``. ``completion_ids`` is ragged with ``C``
    rows and maximum width ``T_max``. Returned tensors have shape
    ``(C, T_max)``. Padding is excluded by both masks; ``decision_mask`` also
    excludes grammar nodes with only one legal child.
    """

    prompt = [int(value) for value in prompt_ids]
    completions = [[int(value) for value in row] for row in completion_ids]
    if not prompt or not completions or any(not row for row in completions):
        raise ValueError("Prompt and every completion must be non-empty.")
    maximum = max(len(row) for row in completions)
    pad_id = int(grammar.eos_token_id)
    rows = [prompt + row + [pad_id] * (maximum - len(row)) for row in completions]
    masks = [
        [1] * (len(prompt) + len(row)) + [0] * (maximum - len(row))
        for row in completions
    ]
    input_ids = torch.tensor(rows, dtype=torch.long, device=device)
    attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
    causal_lm = unwrap_causal_lm(model)
    hidden = _forward_hidden(
        model,
        causal_lm,
        input_ids=input_ids,
        attention_mask=attention_mask,
    )
    prediction_hidden = completion_prediction_hidden(
        hidden,
        prompt_length=len(prompt),
        completion_length=maximum,
    )

    logp_rows: list[torch.Tensor] = []
    valid_rows: list[torch.Tensor] = []
    decision_rows: list[torch.Tensor] = []
    for row_index, completion in enumerate(completions):
        prefix: list[int] = []
        values: list[torch.Tensor] = []
        decisions: list[bool] = []
        for token_index, target_id in enumerate(completion):
            allowed = grammar.allowed_next(prefix)
            if target_id not in allowed:
                raise ValueError(
                    f"Completion row {row_index} token {token_index} is illegal."
                )
            legal_logits = legal_logits_from_hidden(
                causal_lm,
                prediction_hidden[row_index, token_index],
                allowed,
            )
            values.append(
                legal_token_log_prob(
                    legal_logits,
                    list(range(len(allowed))),
                    allowed.index(target_id),
                )
            )
            decisions.append(len(allowed) > 1)
            prefix.append(target_id)
        grammar.parse(prefix)
        padding = maximum - len(completion)
        logp_rows.append(F.pad(torch.stack(values), (0, padding)))
        valid_rows.append(
            torch.tensor(
                [True] * len(completion) + [False] * padding,
                dtype=torch.bool,
                device=device,
            )
        )
        decision_rows.append(
            torch.tensor(
                decisions + [False] * padding,
                dtype=torch.bool,
                device=device,
            )
        )
    return CompletionScores(
        log_probs=torch.stack(logp_rows),
        valid_mask=torch.stack(valid_rows),
        decision_mask=torch.stack(decision_rows),
    )
