"""Isolated RLOO with DAPO-style sampling, clipping, and cached rollout."""

from .config import approved_config, load_config, validate_config
from .objective import (
    ClippedChunkOutput,
    RewardOutput,
    clipped_rloo_token_sum,
    is_effective_rewards,
    rewards_and_advantages,
)
from .rollout import (
    CacheRolloutStats,
    CanonicalCandidate,
    CanonicalPromptRollout,
    PromptRequest,
    PromptRollout,
    rollout_prompt_batch,
)
from .trainer import (
    WindowLRScheduler,
    collect_effective_window,
    learning_rate_for_window,
    train_complete_window,
)

__all__ = [
    "ClippedChunkOutput",
    "CacheRolloutStats",
    "CanonicalCandidate",
    "CanonicalPromptRollout",
    "PromptRequest",
    "PromptRollout",
    "RewardOutput",
    "WindowLRScheduler",
    "approved_config",
    "clipped_rloo_token_sum",
    "collect_effective_window",
    "is_effective_rewards",
    "load_config",
    "learning_rate_for_window",
    "rewards_and_advantages",
    "rollout_prompt_batch",
    "train_complete_window",
    "validate_config",
]
