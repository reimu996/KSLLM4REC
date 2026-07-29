"""Unified legal-action probability primitives for GRPO V3.1."""

from __future__ import annotations

from collections.abc import Sequence

import torch


def legal_log_probs(logits: torch.Tensor, allowed_ids: Sequence[int]) -> torch.Tensor:
    """Return FP32 log-probabilities normalized only over legal token IDs."""

    if logits.ndim != 1:
        raise ValueError("logits must have shape (V,).")
    if not allowed_ids:
        raise ValueError("allowed_ids must not be empty.")
    ids = torch.as_tensor(allowed_ids, dtype=torch.long, device=logits.device)
    if ids.ndim != 1 or int(torch.unique(ids).numel()) != int(ids.numel()):
        raise ValueError("allowed_ids must be a one-dimensional unique set.")
    if int(ids.min()) < 0 or int(ids.max()) >= logits.numel():
        raise ValueError("allowed_ids contains an out-of-vocabulary token.")
    legal_logits = logits.float().index_select(0, ids)
    return legal_logits - torch.logsumexp(legal_logits, dim=0)


def legal_token_log_prob(
    logits: torch.Tensor, allowed_ids: Sequence[int], token_id: int
) -> torch.Tensor:
    """Return one legal token's FP32 log-probability; reject illegal tokens."""

    try:
        position = list(allowed_ids).index(int(token_id))
    except ValueError as exc:
        raise ValueError(f"Token {token_id} is outside the legal action set.") from exc
    return legal_log_probs(logits, allowed_ids)[position]


def sample_legal_token(
    logits: torch.Tensor,
    allowed_ids: Sequence[int],
    *,
    generator: torch.Generator,
) -> tuple[int, torch.Tensor]:
    """Sample one legal token at temperature 1 and return token plus old logp."""

    log_probs = legal_log_probs(logits, allowed_ids)
    position = int(torch.multinomial(log_probs.exp(), 1, generator=generator).item())
    return int(allowed_ids[position]), log_probs[position]
