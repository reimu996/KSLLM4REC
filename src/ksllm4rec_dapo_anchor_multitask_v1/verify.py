"""Deterministic structural verifiers for DAPO-Anchor-Multitask V1."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any, Iterable, Mapping, Sequence

from . import contract
from ._infra.orpo_data import Sid
from .reward import ObjectiveRoute
from .source_blocks import build_source_blocks


@dataclass(frozen=True)
class SourceVerificationResult:
    source_groups: int
    candidate_rollouts: int
    completed_source_blocks: int
    rl_memberships: tuple[tuple[int, str, str], ...]
    anchor_memberships: tuple[tuple[int, str, str], ...]


@dataclass(frozen=True)
class WindowVerificationResult:
    optimization_windows: int
    policy_optimizer_steps: int
    rl_groups: int
    anchor_groups: int
    policy_memberships: tuple[tuple[int, str, str], ...]
    anchor_memberships: tuple[tuple[int, str, str], ...]
    next_source_block: int


@dataclass(frozen=True)
class RunVerificationResult:
    source: SourceVerificationResult
    windows: WindowVerificationResult


def validate_source_group_rows(
    rows: Iterable[Mapping[str, Any]], config: Mapping[str, Any]
) -> SourceVerificationResult:
    """Verify one complete arm-specific pass over the 532 floor blocks."""

    text_enabled = bool(
        config.get("experiments", {}).get("active", {}).get("text_to_sid_enabled")
    )
    expected: list[tuple[int, str, int]] = []
    for block in build_source_blocks(
        recommendation_groups=int(config["data"]["recommendation_groups"]),
        text_to_sid_groups=int(config["data"]["text_to_sid_groups"]),
        blocks=int(config["data"]["source_blocks"]),
    ):
        expected.extend(
            (block.index, "recommendation", source_index)
            for source_index in range(
                block.recommendation.start, block.recommendation.stop
            )
        )
        if text_enabled:
            expected.extend(
                (block.index, "item_text_to_sid", source_index)
                for source_index in range(
                    block.text_to_sid.start, block.text_to_sid.stop
                )
            )
    values = list(rows)
    observed: list[tuple[int, str, int]] = []
    identities: set[tuple[str, str]] = set()
    rl_memberships: list[tuple[int, str, str]] = []
    anchor_memberships: list[tuple[int, str, str]] = []
    previous_policy_step = 0
    for index, row in enumerate(values):
        try:
            identity = (str(row["task"]), str(row["group_id"]))
            triple = (
                int(row["block_index"]),
                identity[0],
                int(row["source_index"]),
            )
            rollout_count = int(row["rollout_count"])
            policy_step = int(row["policy_step_at_sample"])
            route = ObjectiveRoute(str(row["route"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Invalid source row {index}.") from exc
        if rollout_count != contract.GROUP_SIZE:
            raise ValueError(
                f"Source row {index} must contain {contract.GROUP_SIZE} rollouts."
            )
        if policy_step < 0:
            raise ValueError(f"Source row {index} has a negative policy step.")
        if policy_step < previous_policy_step:
            raise ValueError("Source policy-step snapshots must be non-decreasing.")
        previous_policy_step = policy_step
        if identity in identities:
            raise ValueError("Source group identity appears more than once in the source plan.")
        identities.add(identity)
        observed.append(triple)
        membership = (triple[0], identity[0], identity[1])
        if route in {ObjectiveRoute.RL, ObjectiveRoute.RL_AND_ANCHOR}:
            rl_memberships.append(membership)
        if route in {ObjectiveRoute.ANCHOR, ObjectiveRoute.RL_AND_ANCHOR}:
            anchor_memberships.append(membership)
    if observed != expected:
        raise ValueError(
            "Observed source plan differs from the expected 532 floor-partition plan."
        )
    return SourceVerificationResult(
        source_groups=len(values),
        candidate_rollouts=len(values) * contract.GROUP_SIZE,
        completed_source_blocks=int(config["data"]["source_blocks"]),
        rl_memberships=tuple(rl_memberships),
        anchor_memberships=tuple(anchor_memberships),
    )


def _finite(value: Any, *, name: str, non_negative: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric.") from exc
    if not math.isfinite(number) or (non_negative and number < 0.0):
        qualifier = "finite and non-negative" if non_negative else "finite"
        raise ValueError(f"{name} must be {qualifier}.")
    return number


def validate_window_rows(
    rows: Iterable[Mapping[str, Any]], config: Mapping[str, Any]
) -> WindowVerificationResult:
    """Verify K=1 policy minibatches and the merged Anchor gradient budget."""

    low = _finite(config["loss"]["clip_ratio_low"], name="clip low")
    high = _finite(config["loss"]["clip_ratio_high"], name="clip high")
    if low != contract.CLIP_RATIO_LOW or high != contract.CLIP_RATIO_HIGH:
        raise ValueError("Asymmetric clip bounds must be [0.8, 1.28].")
    max_batch = int(config["loss"]["minibatch_groups"])
    target_ratio = float(config["anchor"]["target_gradient_ratio"])
    max_anchor_weight = float(config["anchor"]["max_weight"])
    max_policy_replay = float(
        config["rollout"]["canonical_replay_max_logp_difference"]
    )
    max_anchor_replay = float(config["anchor"]["replay_max_logp_difference"])
    values = list(rows)
    seen_groups: set[tuple[str, str]] = set()
    total_steps = 0
    total_rl_groups = 0
    total_anchor_groups = 0
    policy_memberships: list[tuple[int, str, str]] = []
    anchor_memberships: list[tuple[int, str, str]] = []
    seen_anchor_groups: set[tuple[str, str]] = set()
    expected_policy_cursor = 0
    previous_next_source_block = 0
    for row_index, row in enumerate(values):
        if int(row.get("optimization_window_index", -1)) != row_index:
            raise ValueError("Optimization window indices must be contiguous from zero.")
        first_block = int(row.get("first_source_block", -1))
        next_block = int(row.get("next_source_block", -1))
        if first_block != previous_next_source_block or next_block <= first_block:
            raise ValueError("Optimization window source-block ranges are invalid.")
        previous_next_source_block = next_block
        if int(row.get("k", -1)) != 1:
            raise ValueError("K=1 is required for every optimization window.")
        if int(row.get("anchor_optimizer_steps", -1)) != 0:
            raise ValueError("Anchor must not use an independent optimizer step.")
        policy_steps = int(row.get("policy_optimizer_steps", -1))
        steps = row.get("steps")
        if not isinstance(steps, list) or policy_steps <= 0 or len(steps) != policy_steps:
            raise ValueError("policy_optimizer_steps must equal the logged minibatches.")
        snapshot = int(row.get("snapshot_policy_step", -1))
        completed = int(row.get("completed_policy_steps", -1))
        if snapshot != expected_policy_cursor or completed != snapshot + policy_steps:
            raise ValueError("Policy-step cursors are inconsistent across windows.")
        expected_policy_cursor = completed

        norms_value = row.get("policy_gradient_norms")
        if not isinstance(norms_value, list) or len(norms_value) != policy_steps:
            raise ValueError("Every policy step needs one raw gradient norm.")
        norms = [
            _finite(value, name="policy gradient norm", non_negative=True)
            for value in norms_value
        ]
        reference = _finite(
            row.get("rl_reference_gradient_norm"),
            name="RL reference gradient norm",
            non_negative=True,
        )
        if not math.isclose(reference, float(median(norms)), rel_tol=1.0e-6, abs_tol=1.0e-12):
            raise ValueError("RL reference gradient norm must be the policy-step median.")

        window_groups: list[tuple[str, str]] = []
        for step_index, step in enumerate(steps):
            if not isinstance(step, Mapping):
                raise ValueError("Each policy step must be a mapping.")
            ids = step.get("group_ids")
            tasks = step.get("tasks")
            if not isinstance(ids, (list, tuple)) or not isinstance(tasks, (list, tuple)):
                raise ValueError("Every policy step must log group_ids and tasks.")
            if not 1 <= len(ids) <= max_batch or len(tasks) != len(ids):
                raise ValueError("Policy minibatches must contain 1..8 task/group pairs.")
            if step_index + 1 < policy_steps and len(ids) != max_batch:
                raise ValueError("Only the final policy minibatch may contain fewer than 8 groups.")
            identities = [
                (str(task), str(group_id))
                for task, group_id in zip(tasks, ids, strict=True)
            ]
            if any(identity in seen_groups for identity in identities) or len(
                set(identities)
            ) != len(identities):
                raise ValueError("K=1 forbids reusing a policy group.")
            seen_groups.update(identities)
            window_groups.extend(identities)
            policy_memberships.extend(
                (row_index, task, group_id) for task, group_id in identities
            )
            decision_tokens = int(step.get("decision_tokens", -1))
            clipped = int(step.get("clipped_tokens", -1))
            below = int(step.get("below_low_tokens", -1))
            above = int(step.get("above_high_tokens", -1))
            if min(decision_tokens, clipped, below, above) < 0:
                raise ValueError("Policy token counters must be non-negative.")
            if clipped > below + above or below + above > decision_tokens:
                raise ValueError("Clip token counters are inconsistent.")
            ratio_min = _finite(step.get("ratio_min"), name="ratio_min")
            ratio_mean = _finite(step.get("ratio_mean"), name="ratio_mean")
            ratio_max = _finite(step.get("ratio_max"), name="ratio_max")
            if not 0.0 < ratio_min <= ratio_mean <= ratio_max:
                raise ValueError("Policy ratios must be positive and ordered.")
            if ratio_min < low and below == 0:
                raise ValueError("A ratio below 0.8 must increment below_low_tokens.")
            if ratio_max > high and above == 0:
                raise ValueError("A ratio above 1.28 must increment above_high_tokens.")
            for name in ("loss", "learning_rate", "max_replay_logp_difference"):
                number = _finite(step.get(name), name=name, non_negative=name != "loss")
                if name == "learning_rate":
                    train = config["train"]
                    policy_step = snapshot + step_index + 1
                    per_level = int(train["warmup_steps_per_level"])
                    levels = int(train["warmup_policy_steps"]) // per_level
                    level = min((policy_step - 1) // per_level + 1, levels)
                    expected_lr = float(train["learning_rate"]) * level / levels
                    if not math.isclose(
                        number, expected_lr, rel_tol=1.0e-9, abs_tol=1.0e-15
                    ):
                        raise ValueError("Policy-step learning rate differs from the schedule.")
                if (
                    name == "max_replay_logp_difference"
                    and step_index == 0
                    and number > max_policy_replay
                ):
                    raise ValueError(
                        "The first policy minibatch replay logp difference exceeds "
                        "the frozen-policy gate."
                    )
        if int(row.get("rl_groups", -1)) != len(window_groups):
            raise ValueError("rl_groups differs from the policy minibatch membership.")

        raw = _finite(
            row.get("anchor_raw_gradient_norm"),
            name="Anchor raw gradient norm",
            non_negative=True,
        )
        scaled = _finite(
            row.get("anchor_scaled_gradient_norm"),
            name="Anchor scaled gradient norm",
            non_negative=True,
        )
        cap = _finite(row.get("lambda_cap"), name="lambda_cap", non_negative=True)
        effective = _finite(
            row.get("lambda_effective"), name="lambda_effective", non_negative=True
        )
        ratio = _finite(
            row.get("anchor_to_rl_gradient_ratio"),
            name="Anchor/RL gradient ratio",
            non_negative=True,
        )
        _finite(row.get("combined_gradient_norm"), name="combined gradient norm", non_negative=True)
        if effective > min(cap, max_anchor_weight) + 1.0e-12:
            raise ValueError("lambda_effective exceeds its calibrated/cap limit.")
        if not math.isclose(scaled, raw * effective, rel_tol=1.0e-6, abs_tol=1.0e-12):
            raise ValueError("Scaled Anchor gradient norm is inconsistent.")
        if reference > 0.0 and raw > 0.0:
            expected_cap = target_ratio * reference / raw
            if not math.isclose(cap, expected_cap, rel_tol=1.0e-6, abs_tol=1.0e-12):
                raise ValueError("lambda_cap is inconsistent with the 10% budget.")
            if not math.isclose(ratio, scaled / reference, rel_tol=1.0e-6, abs_tol=1.0e-12):
                raise ValueError("Anchor/RL gradient ratio is inconsistent.")
        elif any(value != 0.0 for value in (cap, effective, scaled, ratio)):
            raise ValueError("Zero RL or Anchor gradient must disable the Anchor merge.")
        if ratio > target_ratio + 1.0e-12:
            raise ValueError("Scaled Anchor gradient exceeds 10% of the RL reference gradient.")
        anchor_results = row.get("anchor_group_results")
        anchor_count = int(row.get("anchor_groups", -1))
        if not isinstance(anchor_results, list) or len(anchor_results) != anchor_count:
            raise ValueError("anchor_groups differs from anchor_group_results.")
        anchor_ids = [
            (str(item.get("task")), str(item.get("group_id")))
            for item in anchor_results
            if isinstance(item, Mapping)
        ]
        if len(anchor_ids) != anchor_count or len(set(anchor_ids)) != anchor_count:
            raise ValueError("Anchor group results must contain unique task/group pairs.")
        if any(identity in seen_anchor_groups for identity in anchor_ids):
            raise ValueError("A source group entered Anchor more than once.")
        seen_anchor_groups.update(anchor_ids)
        anchor_memberships.extend(
            (row_index, task, group_id) for task, group_id in anchor_ids
        )
        for item in anchor_results:
            if not isinstance(item, Mapping):
                raise ValueError("Every Anchor group result must be a mapping.")
            _finite(item.get("set_nll"), name="Anchor set NLL", non_negative=True)
            _finite(
                item.get("all_gt_decision_nll_sum"),
                name="Anchor all-GT decision NLL",
                non_negative=True,
            )
            if int(item.get("gt_sid_count", 0)) <= 0 or int(
                item.get("decision_tokens", 0)
            ) <= 0:
                raise ValueError("Every Anchor group must score GT decision tokens.")
        replay = _finite(
            row.get("anchor_max_replay_logp_difference"),
            name="Anchor replay logp difference",
            non_negative=True,
        )
        if replay > max_anchor_replay:
            raise ValueError("Anchor replay logp difference exceeds the frozen gate.")
        total_steps += policy_steps
        total_rl_groups += len(window_groups)
        total_anchor_groups += anchor_count
    return WindowVerificationResult(
        optimization_windows=len(values),
        policy_optimizer_steps=total_steps,
        rl_groups=total_rl_groups,
        anchor_groups=total_anchor_groups,
        policy_memberships=tuple(policy_memberships),
        anchor_memberships=tuple(anchor_memberships),
        next_source_block=previous_next_source_block,
    )


def _validate_group_rows(
    rows: Iterable[Mapping[str, Any]], windows: WindowVerificationResult, config: Mapping[str, Any]
) -> int:
    values = list(rows)
    observed: list[tuple[int, str, str]] = []
    allowed_rewards = {float(value) for value in config["reward"].values()}
    for index, row in enumerate(values):
        window_index = int(row.get("optimization_window_index", -1))
        identity = (window_index, str(row.get("task")), str(row.get("group_id")))
        if window_index < 0 or int(row.get("source_block", -1)) < 0 or int(
            row.get("source_index", -1)
        ) < 0:
            raise ValueError(f"RL group row {index} has invalid source counters.")
        if int(row.get("k", -1)) != 1:
            raise ValueError("RL group log violates K=1.")
        if int(row.get("gt_injection_count", -1)) != 0:
            raise ValueError("RL candidates must not contain GT injection.")
        sequences = [
            row.get("rewards"),
            row.get("advantages"),
            row.get("tiers"),
            row.get("candidate_sids"),
        ]
        if any(not isinstance(value, list) or len(value) != contract.GROUP_SIZE for value in sequences):
            raise ValueError("Every RL group must retain all 16 candidate records.")
        rewards = [_finite(value, name="reward") for value in row["rewards"]]
        advantages = [_finite(value, name="advantage") for value in row["advantages"]]
        if any(
            not any(
                math.isclose(value, allowed, rel_tol=0.0, abs_tol=1.0e-6)
                for allowed in allowed_rewards
            )
            for value in rewards
        ):
            raise ValueError("RL group contains a reward outside the active arm table.")
        if not math.isclose(sum(advantages), 0.0, abs_tol=1.0e-5):
            raise ValueError("RLOO group advantages must sum to zero.")
        observed.append(identity)
    if len(set(observed)) != len(observed):
        raise ValueError("groups.jsonl repeats a K=1 policy group.")
    if set(observed) != set(windows.policy_memberships):
        raise ValueError("groups.jsonl membership differs from windows.jsonl.")
    return len(values)


def validate_run_summary(
    summary: Mapping[str, Any],
    *,
    source: SourceVerificationResult,
    windows: WindowVerificationResult,
    group_rows: Iterable[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> None:
    """Recompute every run-summary counter from source, group, and window logs."""

    rl_groups = _validate_group_rows(group_rows, windows, config)
    source_rl = {(task, group_id) for _, task, group_id in source.rl_memberships}
    window_rl = {(task, group_id) for _, task, group_id in windows.policy_memberships}
    if source_rl != window_rl:
        raise ValueError("Source routes and RL optimization membership differ.")
    tail_start = windows.next_source_block
    source_anchor_before_tail = {
        (task, group_id)
        for block, task, group_id in source.anchor_memberships
        if block < tail_start
    }
    source_anchor_tail = {
        (task, group_id)
        for block, task, group_id in source.anchor_memberships
        if block >= tail_start
    }
    window_anchor = {
        (task, group_id) for _, task, group_id in windows.anchor_memberships
    }
    if source_anchor_before_tail != window_anchor:
        raise ValueError("Source routes and merged Anchor membership differ.")
    expected = {
        "spec_version": contract.SPEC_VERSION,
        "arm": config["experiments"]["active"]["arm"],
        "completed_source_blocks": source.completed_source_blocks,
        "source_groups": source.source_groups,
        "candidate_rollouts": source.candidate_rollouts,
        "optimization_windows": windows.optimization_windows,
        "completed_policy_steps": windows.policy_optimizer_steps,
        "rl_groups": rl_groups,
        "complete": True,
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise ValueError(f"run_summary.{key} differs from recomputed logs.")
    final_flush = int(summary.get("final_auxiliary_flush_steps", -1))
    if final_flush not in (0, 1):
        raise ValueError("The source-end auxiliary flush permits zero or one optimizer step.")
    if int(summary.get("total_optimizer_steps", -1)) != (
        windows.policy_optimizer_steps + final_flush
    ):
        raise ValueError("run_summary.total_optimizer_steps is inconsistent.")
    auxiliary = summary.get("final_auxiliary_flush")
    if not source_anchor_tail:
        if auxiliary is not None or final_flush != 0:
            raise ValueError("A run without trailing Anchor groups cannot flush them.")
    else:
        if not isinstance(auxiliary, Mapping):
            raise ValueError("Trailing Anchor groups require an auxiliary audit record.")
        required_auxiliary = {
            "first_source_block",
            "next_source_block",
            "snapshot_policy_step",
            "anchor_groups",
            "gt_sid_rows",
            "decision_tokens",
            "optimizer_steps",
            "scheduler_advanced",
            "rl_reference_gradient_norm",
            "lambda_cap",
            "lambda_effective",
            "anchor_raw_gradient_norm",
            "anchor_scaled_gradient_norm",
            "anchor_to_rl_gradient_ratio",
            "combined_gradient_norm",
            "anchor_set_nll_mean",
            "anchor_max_replay_logp_difference",
        }
        if set(auxiliary) != required_auxiliary:
            raise ValueError("Final auxiliary audit has an invalid schema.")
        if (
            int(auxiliary["first_source_block"]) != tail_start
            or int(auxiliary["next_source_block"]) != source.completed_source_blocks
            or int(auxiliary["snapshot_policy_step"])
            != windows.policy_optimizer_steps
            or int(auxiliary["anchor_groups"]) != len(source_anchor_tail)
            or int(auxiliary["gt_sid_rows"]) < len(source_anchor_tail)
            or int(auxiliary["decision_tokens"]) <= 0
            or auxiliary["scheduler_advanced"] is not False
            or int(auxiliary["optimizer_steps"]) != final_flush
        ):
            raise ValueError("Final auxiliary audit counters are inconsistent.")
        reference = _finite(
            auxiliary["rl_reference_gradient_norm"],
            name="Final auxiliary RL reference",
            non_negative=True,
        )
        raw = _finite(
            auxiliary["anchor_raw_gradient_norm"],
            name="Final auxiliary raw gradient",
            non_negative=True,
        )
        scaled = _finite(
            auxiliary["anchor_scaled_gradient_norm"],
            name="Final auxiliary scaled gradient",
            non_negative=True,
        )
        cap = _finite(auxiliary["lambda_cap"], name="Final auxiliary cap", non_negative=True)
        effective = _finite(
            auxiliary["lambda_effective"],
            name="Final auxiliary effective weight",
            non_negative=True,
        )
        ratio = _finite(
            auxiliary["anchor_to_rl_gradient_ratio"],
            name="Final auxiliary gradient ratio",
            non_negative=True,
        )
        _finite(
            auxiliary["combined_gradient_norm"],
            name="Final auxiliary combined gradient",
            non_negative=True,
        )
        _finite(
            auxiliary["anchor_set_nll_mean"],
            name="Final auxiliary set NLL",
            non_negative=True,
        )
        replay = _finite(
            auxiliary["anchor_max_replay_logp_difference"],
            name="Final auxiliary replay difference",
            non_negative=True,
        )
        if replay > float(config["anchor"]["replay_max_logp_difference"]):
            raise ValueError("Final auxiliary replay difference exceeds the gate.")
        max_weight = float(config["anchor"]["max_weight"])
        target_ratio = float(config["anchor"]["target_gradient_ratio"])
        if effective > min(cap, max_weight) + 1.0e-12:
            raise ValueError("Final auxiliary effective weight exceeds its budget.")
        if not math.isclose(scaled, raw * effective, rel_tol=1.0e-6, abs_tol=1.0e-12):
            raise ValueError("Final auxiliary scaled gradient is inconsistent.")
        if reference > 0.0 and raw > 0.0:
            if not math.isclose(
                cap,
                target_ratio * reference / raw,
                rel_tol=1.0e-6,
                abs_tol=1.0e-12,
            ):
                raise ValueError("Final auxiliary lambda cap is inconsistent.")
            if not math.isclose(
                ratio, scaled / reference, rel_tol=1.0e-6, abs_tol=1.0e-12
            ):
                raise ValueError("Final auxiliary gradient ratio is inconsistent.")
        elif any(value != 0.0 for value in (cap, effective, scaled, ratio)):
            raise ValueError("Zero reference/raw gradient must disable the final flush.")
        if ratio > target_ratio + 1.0e-12:
            raise ValueError("Final auxiliary gradient exceeds the 10% budget.")
        expected_steps = int(effective > 0.0)
        if final_flush != expected_steps:
            raise ValueError("Final auxiliary optimizer-step decision is inconsistent.")
    if not summary.get("last_checkpoint"):
        raise ValueError("A complete run must identify its final checkpoint.")


_RUN_TOP_LEVEL_ALLOWLIST = {
    "resolved_config.json",
    "runtime_signature.json",
    "source_groups.jsonl",
    "groups.jsonl",
    "windows.jsonl",
    "run_summary.json",
    "recovery",
    "source-025-adapter",
    "source-050-adapter",
    "source-075-adapter",
    "source-100-adapter",
}


def validate_run_directory_contents(run_dir: str | Path) -> None:
    """Reject monitoring files and every other uncontracted top-level artifact."""

    root = Path(run_dir)
    if not root.is_dir():
        raise FileNotFoundError(root)
    unexpected = sorted(path.name for path in root.iterdir() if path.name not in _RUN_TOP_LEVEL_ALLOWLIST)
    if unexpected:
        raise ValueError(f"Run directory contains uncontracted artifacts: {unexpected}")


def _recursive_difference_paths(
    left: Any, right: Any, prefix: tuple[str, ...] = ()
) -> set[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        result: set[str] = set()
        for key in sorted(set(left) | set(right), key=str):
            path = (*prefix, str(key))
            if key not in left or key not in right:
                result.add(".".join(path))
            else:
                result.update(_recursive_difference_paths(left[key], right[key], path))
        return result
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        result = set()
        for index in range(max(len(left), len(right))):
            path = (*prefix, str(index))
            if index >= len(left) or index >= len(right):
                result.add(".".join(path))
            else:
                result.update(_recursive_difference_paths(left[index], right[index], path))
        return result
    return set() if left == right else {".".join(prefix)}


def validate_experiment_matrix(
    configs: Mapping[str, Mapping[str, Any]],
) -> dict[str, set[str]]:
    """Permit only the confirmed A/B/C reward and task-factor differences."""

    if set(configs) != set(contract.EXPERIMENT_ARMS):
        raise ValueError("Experiment matrix must contain exactly arms A, B, and C.")
    for arm in contract.EXPERIMENT_ARMS:
        active = configs[arm].get("experiments", {}).get("active", {})
        if active.get("arm") != arm:
            raise ValueError(f"Experiment matrix key {arm} contradicts active.arm.")
    allowed = {
        "A:B": {
            "experiments.active.arm",
            "experiments.active.reward_profile",
            "reward.same_a",
            "reward.same_ab",
        },
        "B:C": {
            "experiments.active.arm",
            "experiments.active.text_to_sid_enabled",
        },
        "A:C": {
            "experiments.active.arm",
            "experiments.active.reward_profile",
            "experiments.active.text_to_sid_enabled",
            "reward.same_a",
            "reward.same_ab",
        },
    }
    isolation_paths = {
        "anchor.calibration_path",
        "output.log_dir",
        "output.run_dir",
    }
    result: dict[str, set[str]] = {}
    for pair, expected in allowed.items():
        left, right = pair.split(":")
        actual = _recursive_difference_paths(configs[left], configs[right])
        missing_isolation = isolation_paths - actual
        if missing_isolation:
            raise ValueError(
                "Experiment arms must use different run/log/calibration paths: "
                f"{sorted(missing_isolation)[0]}"
            )
        actual -= isolation_paths
        if actual != expected:
            unexpected = sorted(actual - expected)
            missing = sorted(expected - actual)
            detail = unexpected[0] if unexpected else missing[0]
            raise ValueError(
                f"Experiment pair {pair} has an unconfirmed or missing difference: {detail}."
            )
        result[pair] = actual
    return result


def validate_probe64_report(
    report: Mapping[str, Any],
    *,
    expected_rows: Sequence[Mapping[str, Any]],
    trie: Any,
    require_complete: bool,
) -> int:
    """Bind every reported beam candidate to the fixed probe and SID trie."""

    if int(report.get("schema_version", -1)) != 1:
        raise ValueError("Probe report must use schema_version=1.")
    if report.get("spec_version") != contract.SPEC_VERSION:
        raise ValueError("Probe report belongs to another spec.")
    if int(report.get("num_candidates", -1)) != contract.EVALUATION_CANDIDATES:
        raise ValueError("Probe report must declare num_candidates=64.")
    if report.get("require_unique_candidates") is not True:
        raise ValueError("Probe report must require unique candidates.")
    if report.get("cache_implementation") != "offloaded":
        raise ValueError("Probe report must use the exact offloaded KV cache.")

    rows = tuple(expected_rows)
    selected = report.get("selected_indices")
    results = report.get("results")
    if not isinstance(selected, list) or not isinstance(results, list):
        raise ValueError("Probe report must contain selected_indices and results lists.")
    try:
        indices = tuple(int(value) for value in selected)
    except (TypeError, ValueError) as exc:
        raise ValueError("Probe selected_indices must be integers.") from exc
    if not indices or tuple(sorted(set(indices))) != indices:
        raise ValueError("Probe selected_indices must be non-empty, unique, and sorted.")
    if indices[0] < 0 or indices[-1] >= len(rows):
        raise ValueError("Probe selected_indices exceed the fixed probe.")
    if len(results) != len(indices) or int(report.get("result_count", -1)) != len(results):
        raise ValueError("Probe result_count differs from selected_indices.")
    complete = indices == tuple(range(len(rows)))
    if report.get("complete") is not complete:
        raise ValueError("Probe complete flag differs from selected_indices.")
    if require_complete and not complete:
        raise ValueError("A complete Probe64 report must cover every fixed row.")

    result_fields = {
        "probe_index",
        "task",
        "source_row_sha256",
        "source_line",
        "target_sid",
        "prompt_tokens",
        "candidate_sids",
        "hit_rank",
        "pass_at_64",
    }
    pass_count = 0
    for position, (probe_index, result) in enumerate(
        zip(indices, results, strict=True)
    ):
        expected = rows[probe_index]
        if not isinstance(result, Mapping) or set(result) != result_fields:
            raise ValueError(f"Probe result {position} has an invalid schema.")
        expected_identity = {
            "probe_index": probe_index,
            "task": str(expected["task"]),
            "source_row_sha256": str(expected["source_row_sha256"]),
            "source_line": int(expected["source_line"]),
            "target_sid": str(expected["target_sid"]),
        }
        observed_identity = {
            key: int(result[key]) if key in {"probe_index", "source_line"} else str(result[key])
            for key in expected_identity
        }
        if observed_identity != expected_identity:
            raise ValueError(f"Probe result {position} differs from its fixed source row.")
        if int(result["prompt_tokens"]) <= 0:
            raise ValueError(f"Probe result {position} has no prompt tokens.")
        candidates = result["candidate_sids"]
        if not isinstance(candidates, list) or len(candidates) != 64:
            raise ValueError(f"Probe result {position} must contain 64 unique legal SIDs.")
        parsed: list[Sid] = []
        try:
            for value in candidates:
                sid = Sid.parse(str(value))
                if sid.render() != value or not bool(trie.contains(sid)):
                    raise ValueError(value)
                parsed.append(sid)
            target = Sid.parse(str(result["target_sid"]))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Probe result {position} must contain 64 unique trie SIDs."
            ) from exc
        rendered = [sid.render() for sid in parsed]
        if len(set(rendered)) != 64:
            raise ValueError(f"Probe result {position} must contain 64 unique legal SIDs.")
        hit_rank = next(
            (rank for rank, candidate in enumerate(parsed, start=1) if candidate == target),
            None,
        )
        if result["hit_rank"] != hit_rank:
            raise ValueError(f"Probe result {position} reports the wrong hit_rank.")
        passed = hit_rank is not None
        if result["pass_at_64"] is not passed:
            raise ValueError(f"Probe result {position} reports the wrong pass_at_64 value.")
        pass_count += int(passed)
    if int(report.get("pass_count", -1)) != pass_count:
        raise ValueError("Probe pass_count differs from the recomputed result.")
    return len(results)


def _read_jsonl(path: Path) -> list[Mapping[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    rows: list[Mapping[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path.name}:{line_number} is blank.")
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise ValueError(f"{path.name}:{line_number} is not a JSON object.")
            rows.append(value)
    return rows


def verify_run_directory(
    run_dir: str | Path, config: Mapping[str, Any]
) -> RunVerificationResult:
    """Load and cross-check one complete Multitask V1 run directory."""

    root = Path(run_dir)
    validate_run_directory_contents(root)
    source = validate_source_group_rows(_read_jsonl(root / "source_groups.jsonl"), config)
    windows = validate_window_rows(_read_jsonl(root / "windows.jsonl"), config)
    group_rows = _read_jsonl(root / "groups.jsonl")
    summary_path = root / "run_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(summary_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if not isinstance(summary, Mapping):
        raise ValueError("run_summary.json must contain one JSON object.")
    validate_run_summary(
        summary,
        source=source,
        windows=windows,
        group_rows=group_rows,
        config=config,
    )
    return RunVerificationResult(source=source, windows=windows)


__all__ = [
    "SourceVerificationResult",
    "RunVerificationResult",
    "WindowVerificationResult",
    "validate_experiment_matrix",
    "validate_probe64_report",
    "validate_run_summary",
    "validate_run_directory_contents",
    "validate_source_group_rows",
    "validate_window_rows",
    "verify_run_directory",
]
