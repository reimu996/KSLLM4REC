"""Exact acceptance/rejection correction for cached proposal distributions."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch


@dataclass(frozen=True)
class SpeculativeSelection:
    selected_index: int
    accepted: bool
    acceptance_probability: float


def _validate_uniform(value: float, label: str) -> float:
    result = float(value)
    if not math.isfinite(result) or not 0.0 <= result < 1.0:
        raise ValueError(f"{label} must be finite and in [0,1).")
    return result


def _sample_with_uniform(weights: torch.Tensor, uniform: float) -> int:
    values = weights.detach().float()
    total = values.sum()
    if not bool(torch.isfinite(values).all()) or float(total.item()) <= 0.0:
        raise RuntimeError("Residual distribution has no positive finite mass.")
    threshold = float(uniform) * float(total.item())
    cumulative = torch.cumsum(values, dim=0)
    index = int(torch.searchsorted(cumulative, threshold, right=True).item())
    return min(index, values.numel() - 1)


def speculative_accept_or_residual(
    proposal_log_probs: torch.Tensor,
    target_log_probs: torch.Tensor,
    *,
    proposal_index: int,
    acceptance_uniform: float,
    residual_uniform: float,
) -> SpeculativeSelection:
    """Correct one proposal token so the returned token follows the target."""

    if proposal_log_probs.ndim != 1 or target_log_probs.ndim != 1:
        raise ValueError("Proposal and target log probabilities must be vectors.")
    if proposal_log_probs.shape != target_log_probs.shape:
        raise ValueError("Proposal and target vectors must have the same shape.")
    if proposal_log_probs.numel() <= 1:
        raise ValueError("Speculative correction requires a decision distribution.")
    index = int(proposal_index)
    if isinstance(proposal_index, bool) or not 0 <= index < proposal_log_probs.numel():
        raise ValueError("proposal_index is outside the probability vector.")
    accept_u = _validate_uniform(acceptance_uniform, "acceptance_uniform")
    residual_u = _validate_uniform(residual_uniform, "residual_uniform")
    proposal = proposal_log_probs.detach().float()
    target = target_log_probs.detach().float()
    if not bool(torch.isfinite(proposal).all()) or not bool(torch.isfinite(target).all()):
        raise FloatingPointError("Speculative log probabilities must be finite.")

    log_ratio = float((target[index] - proposal[index]).item())
    acceptance = min(1.0, math.exp(log_ratio))
    if accept_u < acceptance:
        return SpeculativeSelection(index, True, acceptance)

    proposal_probabilities = proposal.exp()
    target_probabilities = target.exp()
    residual = (target_probabilities - proposal_probabilities).clamp_min(0.0)
    selected = _sample_with_uniform(residual, residual_u)
    return SpeculativeSelection(selected, False, acceptance)


__all__ = ["SpeculativeSelection", "speculative_accept_or_residual"]
