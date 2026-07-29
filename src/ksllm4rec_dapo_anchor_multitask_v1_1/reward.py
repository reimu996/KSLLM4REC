"""Shared SID hierarchy rewards and objective routing for both RL tasks."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Mapping, Sequence

from ._infra.orpo_data import Sid


GROUP_SIZE = 16
REWARD_VALUES = MappingProxyType(
    {
        "exact": 1.00,
        "same_ab": 0.40,
        "same_a": 0.15,
        "same_domain": 0.01,
        "other_domain": 0.00,
    }
)


class RewardTier(StrEnum):
    EXACT = "exact"
    SAME_AB = "same_ab"
    SAME_A = "same_a"
    SAME_DOMAIN = "same_domain"
    OTHER_DOMAIN = "other_domain"


class ObjectiveRoute(StrEnum):
    RL_AND_ANCHOR = "rl_and_anchor"
    ANCHOR = "anchor"


@dataclass(frozen=True)
class SidReward:
    value: float
    tier: RewardTier


@dataclass(frozen=True)
class GroupReward:
    rewards: tuple[float, ...]
    tiers: tuple[RewardTier, ...]
    effective: bool
    has_exact: bool

    def __post_init__(self) -> None:
        if len(self.rewards) != GROUP_SIZE or len(self.tiers) != GROUP_SIZE:
            raise ValueError(f"A reward group must contain exactly {GROUP_SIZE} slots.")


@dataclass(frozen=True)
class RouteDecision:
    route: ObjectiveRoute
    use_rl: bool
    use_anchor: bool


def _validate_positives(positives: Sequence[Sid]) -> tuple[Sid, ...]:
    values = tuple(positives)
    if not values:
        raise ValueError("Every SID target group must contain at least one GT SID.")
    if not all(isinstance(value, Sid) for value in values):
        raise TypeError("Every GT must be a Sid.")
    if len(set(values)) != len(values):
        raise ValueError("GT SIDs must be unique within a group.")
    return values


def _reward_values(values: Mapping[str, float] | None) -> Mapping[str, float]:
    selected = REWARD_VALUES if values is None else values
    if set(selected) != set(REWARD_VALUES):
        raise ValueError("Reward profile must define exactly the five SID tiers.")
    return selected


def score_sid(
    candidate: Sid,
    positives: Sequence[Sid],
    *,
    reward_values: Mapping[str, float] | None = None,
) -> SidReward:
    """Return the highest shared hierarchy reward against the accepted GT set."""

    if not isinstance(candidate, Sid):
        raise TypeError("candidate must be a Sid.")
    gt = _validate_positives(positives)
    if candidate in gt:
        tier = RewardTier.EXACT
    elif any(candidate.ab_key == positive.ab_key for positive in gt):
        tier = RewardTier.SAME_AB
    elif any(candidate.a_key == positive.a_key for positive in gt):
        tier = RewardTier.SAME_A
    elif any(candidate.domain == positive.domain for positive in gt):
        tier = RewardTier.SAME_DOMAIN
    else:
        tier = RewardTier.OTHER_DOMAIN
    return SidReward(value=float(_reward_values(reward_values)[tier.value]), tier=tier)


def score_group(
    candidates: Sequence[Sid],
    positives: Sequence[Sid],
    *,
    reward_values: Mapping[str, float] | None = None,
) -> GroupReward:
    values = tuple(candidates)
    if len(values) != GROUP_SIZE:
        raise ValueError(f"Expected exactly {GROUP_SIZE} candidates.")
    selected = _reward_values(reward_values)
    scored = tuple(
        score_sid(candidate, positives, reward_values=selected)
        for candidate in values
    )
    rewards = tuple(item.value for item in scored)
    tiers = tuple(item.tier for item in scored)
    return GroupReward(
        rewards=rewards,
        tiers=tiers,
        effective=max(rewards) > min(rewards),
        has_exact=RewardTier.EXACT in tiers,
    )


def route_group(reward: GroupReward) -> RouteDecision:
    """Give every normalized source group one Anchor contribution per epoch.

    RLOO is meaningful only when the 16 rewards differ.  The GT-set Anchor is
    therefore present for every group, including all-exact and reward-flat
    groups, so source coverage is not silently converted into a skipped loss.
    """

    if reward.effective:
        return RouteDecision(
            ObjectiveRoute.RL_AND_ANCHOR, use_rl=True, use_anchor=True
        )
    return RouteDecision(ObjectiveRoute.ANCHOR, use_rl=False, use_anchor=True)
