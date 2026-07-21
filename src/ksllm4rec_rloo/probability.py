"""Shared legal-prefix probability primitives for the reference-free RLOO run.

The sampler and the teacher-forcing scorer must agree on one probability
definition.  A model's full-vocabulary logits are therefore reduced to the
tokens returned by ``grammar.allowed_next(prefix)`` and normalized in FP32.
Deterministic grammar nodes are handled explicitly: their only child receives
log-probability zero and no random-number draw is made.
"""

from __future__ import annotations

import operator
from collections.abc import Sequence
from typing import Any

import torch


def _as_unique_ids(allowed_ids: Sequence[int] | torch.Tensor) -> list[int]:
    """Validate and materialize a legal token list without changing its order."""

    if isinstance(allowed_ids, torch.Tensor):
        if allowed_ids.ndim != 1:
            raise ValueError("allowed_ids must be one-dimensional.")
        if allowed_ids.dtype == torch.bool:
            raise TypeError("allowed_ids must contain integer token IDs.")
        values = [int(value) for value in allowed_ids.detach().cpu().tolist()]
    else:
        try:
            raw_values = list(allowed_ids)
            values = [operator.index(value) for value in raw_values]
        except (TypeError, ValueError) as exc:
            raise TypeError("allowed_ids must be a sequence of integers.") from exc
    if not values:
        raise ValueError("allowed_ids must not be empty.")
    # ``int(True)`` is 1, so reject bools before conversion when possible.
    if isinstance(allowed_ids, torch.Tensor):
        pass
    else:
        for value in raw_values:
            if isinstance(value, bool):
                raise TypeError("allowed_ids must contain integer token IDs.")
    if len(set(values)) != len(values):
        raise ValueError("allowed_ids must contain unique token IDs.")
    if any(value < 0 for value in values):
        raise ValueError("allowed_ids must contain non-negative token IDs.")
    return values


def _validate_logits(logits: torch.Tensor) -> None:
    if not isinstance(logits, torch.Tensor) or logits.ndim != 1:
        raise ValueError("logits must have shape (V,).")
    if logits.numel() == 0:
        raise ValueError("logits must contain at least one vocabulary entry.")


def legal_log_probs(
    logits: torch.Tensor,
    allowed_ids: Sequence[int] | torch.Tensor,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return FP32 log-probabilities normalized over legal IDs only.

    ``logits`` may be either a full vocabulary vector or a vector already
    projected to ``len(allowed_ids)`` entries.  The latter form is used by the
    model-facing scorer to avoid materializing the full vocabulary slice.
    Callers using a projected vector must pass IDs ``range(len(logits))``.
    """

    _validate_logits(logits)
    ids = _as_unique_ids(allowed_ids)
    if not isinstance(temperature, (int, float)) or temperature <= 0:
        raise ValueError("temperature must be a positive finite number.")
    if not torch.isfinite(torch.tensor(float(temperature))):
        raise ValueError("temperature must be a positive finite number.")

    # Validate vocabulary bounds even for a singleton node; this does not read
    # logits values and therefore preserves the no-forward/no-RNG contract.
    if max(ids) >= logits.numel():
        raise ValueError("allowed_ids contains an out-of-vocabulary token.")

    if len(ids) == 1:
        # Do not inspect logits or call any random operation at deterministic
        # grammar nodes.  A singleton categorical distribution has log p=0.
        return torch.zeros((1,), dtype=torch.float32, device=logits.device)

    if len(ids) == logits.numel() and ids == list(range(logits.numel())):
        legal_logits = logits.float()
    else:
        legal_logits = logits.float().index_select(
            0, torch.as_tensor(ids, dtype=torch.long, device=logits.device)
        )
    legal_logits = legal_logits / float(temperature)
    if bool(torch.isnan(legal_logits).any()) or not bool(
        torch.isfinite(legal_logits).any()
    ):
        raise ValueError("legal logits contain no finite value.")
    result = legal_logits - torch.logsumexp(legal_logits, dim=0)
    if bool(torch.isnan(result).any()):
        raise ValueError("legal log-probabilities contain NaN.")
    return result


def legal_token_log_prob(
    logits: torch.Tensor,
    allowed_ids: Sequence[int] | torch.Tensor,
    token_id: int,
    *,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Return one target token's probability under the legal denominator."""

    ids = _as_unique_ids(allowed_ids)
    try:
        position = ids.index(int(token_id))
    except ValueError as exc:
        raise ValueError(f"Token {token_id} is outside the legal action set.") from exc
    return legal_log_probs(logits, ids, temperature=temperature)[position]


def is_decision_node(allowed_ids: Sequence[int] | torch.Tensor) -> bool:
    """Whether a grammar node has more than one legal next token."""

    return len(_as_unique_ids(allowed_ids)) > 1


def sample_legal_token(
    logits: torch.Tensor,
    allowed_ids: Sequence[int] | torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    temperature: float = 1.0,
) -> tuple[int, torch.Tensor]:
    """Sample one legal token and return ``(token_id, log_probability)``.

    At a singleton node this returns immediately, so the supplied generator is
    untouched.  At a branching node, sampling uses exactly the same
    ``legal_log_probs`` primitive used by teacher-forcing rescoring.
    """

    _validate_logits(logits)
    ids = _as_unique_ids(allowed_ids)
    if len(ids) == 1:
        return ids[0], torch.zeros((), dtype=torch.float32, device=logits.device)
    log_probs = legal_log_probs(logits, ids, temperature=temperature)
    kwargs: dict[str, Any] = {}
    if generator is not None:
        kwargs["generator"] = generator
    position = int(torch.multinomial(log_probs.exp(), 1, **kwargs).item())
    return ids[position], log_probs[position]


def sample_legal_action(
    logits: torch.Tensor,
    allowed_ids: Sequence[int] | torch.Tensor,
    *,
    generator: torch.Generator | None = None,
    temperature: float = 1.0,
) -> tuple[int, torch.Tensor, bool]:
    """As ``sample_legal_token`` plus the decision-node flag."""

    decision = is_decision_node(allowed_ids)
    token_id, log_prob = sample_legal_token(
        logits,
        allowed_ids,
        generator=generator,
        temperature=temperature,
    )
    return token_id, log_prob, decision


# Explicit aliases make the shared primitive easy to discover in call sites
# and keep compatibility with the historical GRPO naming.
legal_token_log_probability = legal_token_log_prob
sample_legal_action_token = sample_legal_action


__all__ = [
    "is_decision_node",
    "legal_log_probs",
    "legal_token_log_prob",
    "legal_token_log_probability",
    "sample_legal_action",
    "sample_legal_action_token",
    "sample_legal_token",
]
