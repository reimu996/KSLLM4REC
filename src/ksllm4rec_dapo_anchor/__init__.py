"""DAPO-Anchor: GRPO-DAPO training with post-RL GT anchor loss.

Phase 1 — RL (identical to ksllm4rec_rloo_dapo):
  4 optimizer steps, clipped group-relative loss, dynamic sampling.

Phase 2 — Anchor (added):
  For each filter group (reward-flat group collected by dynamic sampling),
  apply teacher-forced GT anchor loss ``-logsumexp(GT_logps)`` with
  a fixed weight ``ANCHOR_WEIGHT = 0.05``.
"""

from __future__ import annotations

from .objective import (
    group_relative_advantages,
    group_relative_rewards_and_advantages,
    gt_set_anchor_loss,
    is_effective_rewards,
)
from .trainer import run_training, WindowLRScheduler

__all__ = [
    "group_relative_advantages",
    "group_relative_rewards_and_advantages",
    "gt_set_anchor_loss",
    "is_effective_rewards",
    "run_training",
    "WindowLRScheduler",
]
