"""Deterministic structural verifiers for the isolated DAPO-Anchor V2.2 run."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from . import contract


def _finite_non_negative(row: Mapping[str, Any], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{key} must be finite and non-negative.")
    return value


def validate_window_log_row(row: Mapping[str, Any]) -> None:
    """Validate one complete window, including the one-step anchor contract."""

    required_keys = {
        "window_index",
        "effective_groups",
        "filter_groups",
        "anchor_candidate_groups",
        "anchor_groups_used",
        "anchor_group_ids",
        "anchor_gt_sid_count",
        "anchor_decision_token_count",
        "anchor_set_nll_mean",
        "anchor_all_gt_decision_nll",
        "anchor_phase_seconds",
        "anchor_raw_grad_norm",
        "anchor_post_scale_grad_norm",
        "rl_reference_grad_norm",
        "anchor_max_weight",
        "lambda_cap",
        "lambda_effective",
        "anchor_to_rl_grad_ratio",
        "anchor_max_replay_logp_difference",
        "rl_optimizer_steps",
        "anchor_optimizer_steps",
        "optimizer_updates",
        "optimizer_update_step",
        "steps",
    }
    missing = required_keys - set(row)
    if missing:
        raise ValueError(f"Window log row missing keys: {sorted(missing)}")

    window_index = int(row["window_index"])
    if window_index < 0:
        raise ValueError("window_index must be non-negative.")
    effective = int(row["effective_groups"])
    if effective != contract.EFFECTIVE_GROUPS_PER_WINDOW:
        raise ValueError(
            f"Window {window_index}: expected {contract.EFFECTIVE_GROUPS_PER_WINDOW} effective groups, got {effective}."
        )
    filter_groups = int(row["filter_groups"])
    candidates = int(row["anchor_candidate_groups"])
    used = int(row["anchor_groups_used"])
    gt_sids = int(row["anchor_gt_sid_count"])
    decision_tokens = int(row["anchor_decision_token_count"])
    if min(filter_groups, candidates, used, gt_sids, decision_tokens) < 0:
        raise ValueError("Anchor counts must be non-negative.")
    if candidates > filter_groups:
        raise ValueError("anchor_candidate_groups cannot exceed reward-flat filter_groups.")
    if used > candidates:
        raise ValueError("anchor_groups_used cannot exceed anchor_candidate_groups.")
    anchor_ids = row["anchor_group_ids"]
    if not isinstance(anchor_ids, list) or len(anchor_ids) != used:
        raise ValueError("anchor_group_ids must contain exactly every used anchor group.")
    if len(set(anchor_ids)) != len(anchor_ids):
        raise ValueError("anchor_group_ids must be unique within one window.")

    for key in (
        "anchor_set_nll_mean",
        "anchor_all_gt_decision_nll",
        "anchor_phase_seconds",
        "anchor_raw_grad_norm",
        "anchor_post_scale_grad_norm",
        "rl_reference_grad_norm",
        "anchor_max_weight",
        "lambda_cap",
        "lambda_effective",
        "anchor_to_rl_grad_ratio",
        "anchor_max_replay_logp_difference",
    ):
        _finite_non_negative(row, key)

    rl_steps = int(row["rl_optimizer_steps"])
    anchor_steps = int(row["anchor_optimizer_steps"])
    updates = int(row["optimizer_updates"])
    if rl_steps != contract.OPTIMIZER_UPDATES_PER_WINDOW:
        raise ValueError("Every complete window must have exactly four RL optimizer steps.")
    if anchor_steps not in (0, 1):
        raise ValueError("A complete window permits at most one anchor optimizer step.")
    if updates != rl_steps + anchor_steps:
        raise ValueError("optimizer_updates must equal RL plus anchor optimizer steps.")
    if int(row["optimizer_update_step"]) < updates:
        raise ValueError("optimizer_update_step cannot be smaller than this window's updates.")
    if candidates > 0 and used != candidates:
        raise ValueError("Every anchor candidate from W0 onward must be measured and used.")
    if anchor_steps == 1:
        if used != candidates:
            raise ValueError("One anchor optimizer step must use every anchor candidate.")
        if used == 0:
            raise ValueError("An anchor optimizer step requires at least one candidate.")
        if float(row["lambda_effective"]) <= 0.0:
            raise ValueError("An anchor optimizer step requires lambda_effective > 0.")
    elif used not in (0, candidates):
        raise ValueError(
            "Without an optimizer step, every anchor candidate must still be "
            "measured when the gradient or lambda requires a skip."
        )

    ratio = float(row["anchor_to_rl_grad_ratio"])
    if ratio > contract.ANCHOR_TARGET_GRADIENT_RATIO + 1.0e-12:
        raise ValueError("anchor_to_rl_grad_ratio exceeds the 10% pre-clip budget.")
    raw = float(row["anchor_raw_grad_norm"])
    scaled = float(row["anchor_post_scale_grad_norm"])
    rl_reference = float(row["rl_reference_grad_norm"])
    max_weight = float(row["anchor_max_weight"])
    cap = float(row["lambda_cap"])
    effective_lambda = float(row["lambda_effective"])
    if raw > 0.0 and rl_reference > 0.0:
        expected_cap = contract.ANCHOR_TARGET_GRADIENT_RATIO * rl_reference / raw
        expected_effective = min(max_weight, expected_cap)
        if not math.isclose(cap, expected_cap, rel_tol=1.0e-6, abs_tol=1.0e-12):
            raise ValueError("lambda_cap is inconsistent with raw anchor/RL gradients.")
        if not math.isclose(
            effective_lambda, expected_effective, rel_tol=1.0e-6, abs_tol=1.0e-12
        ):
            raise ValueError("lambda_effective is not min(max_weight,cap).")
        if not math.isclose(
            scaled, raw * effective_lambda, rel_tol=1.0e-5, abs_tol=1.0e-12
        ):
            raise ValueError("anchor_post_scale_grad_norm is inconsistent.")
        if not math.isclose(
            ratio, scaled / rl_reference, rel_tol=1.0e-5, abs_tol=1.0e-12
        ):
            raise ValueError("anchor_to_rl_grad_ratio is inconsistent.")
    elif any(value != 0.0 for value in (cap, effective_lambda, scaled, ratio)):
        raise ValueError("Zero RL/anchor signal must skip the weighted anchor gradient.")
    if float(row["anchor_max_replay_logp_difference"]) > contract.CANONICAL_REPLAY_MAX_LOGP_DIFF:
        raise ValueError("Anchor no-grad/grad replay exceeds the 1e-5 gate.")

    steps = row["steps"]
    if not isinstance(steps, list) or len(steps) != rl_steps:
        raise ValueError("steps must contain the four RL minibatches.")
    group_ids: list[str] = []
    for step in steps:
        ids = step.get("group_ids") if isinstance(step, Mapping) else None
        if not isinstance(ids, (list, tuple)) or len(ids) != contract.MINIBATCH_GROUPS:
            raise ValueError("Every RL minibatch must contain exactly eight group IDs.")
        group_ids.extend(str(value) for value in ids)
    if len(group_ids) != contract.EFFECTIVE_GROUPS_PER_WINDOW or len(set(group_ids)) != len(group_ids):
        raise ValueError("K=1 requires exactly 32 distinct effective group IDs per window.")


def validate_group_log_row(row: Mapping[str, Any]) -> None:
    required_keys = {
        "window_index",
        "task",
        "group_id",
        "group_key",
        "effective",
        "gt_injection_count",
        "rewards",
    }
    missing = required_keys - set(row)
    if missing:
        raise ValueError(f"Group log row missing keys: {sorted(missing)}")
    if int(row["window_index"]) < 0:
        raise ValueError("group row window_index must be non-negative.")
    if row["task"] not in {"recommendation", "item_text_to_sid"}:
        raise ValueError("group row task is unsupported.")
    if row["group_key"] != f"{row['task']}|{row['group_id']}":
        raise ValueError("group_key must namespace group_id by task.")
    rewards = row["rewards"]
    if not isinstance(rewards, list) or len(rewards) != contract.GROUP_SIZE:
        raise ValueError("Every group row must retain exactly 16 rollout rewards.")
    if not all(math.isfinite(float(value)) for value in rewards):
        raise ValueError("Group rewards must be finite.")
    if int(row["gt_injection_count"]) != 0:
        raise ValueError("V2.2 never injects a GT candidate into an RL rollout.")


def validate_run_summary(
    summary: Mapping[str, Any], *, expected_windows: int | None = None
) -> None:
    required = {
        "completed_windows",
        "rl_optimizer_update_step",
        "anchor_optimizer_update_step",
        "optimizer_update_step",
        "complete",
    }
    missing = required - set(summary)
    if missing:
        raise ValueError(f"Run summary missing keys: {sorted(missing)}")
    completed = int(summary["completed_windows"])
    rl_steps = int(summary["rl_optimizer_update_step"])
    anchor_steps = int(summary["anchor_optimizer_update_step"])
    total_steps = int(summary["optimizer_update_step"])
    if completed < 0 or anchor_steps < 0:
        raise ValueError("Run summary counters must be non-negative.")
    if rl_steps != completed * contract.OPTIMIZER_UPDATES_PER_WINDOW:
        raise ValueError("Run summary RL step counter is inconsistent.")
    if anchor_steps > max(0, completed - contract.ANCHOR_WARMUP_WINDOWS):
        raise ValueError("Run summary has more than one anchor step per eligible window.")
    if total_steps != rl_steps + anchor_steps:
        raise ValueError("Run summary total optimizer step counter is inconsistent.")
    if expected_windows is not None:
        if completed != expected_windows:
            raise ValueError(f"Expected {expected_windows} completed windows, got {completed}.")
        if bool(summary["complete"]) != (completed == expected_windows):
            raise ValueError("complete flag contradicts completed_windows.")


def check_window_log(window_log: Path) -> tuple[int, int, float]:
    """Return count, all-candidate count, and mean GT-set NLL after validation."""

    rows = 0
    candidates = 0
    losses: list[float] = []
    with Path(window_log).open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            validate_window_log_row(row)
            rows += 1
            candidates += int(row["anchor_candidate_groups"])
            losses.append(float(row["anchor_set_nll_mean"]))
    return rows, candidates, sum(losses) / max(len(losses), 1)


__all__ = [
    "check_window_log",
    "validate_group_log_row",
    "validate_run_summary",
    "validate_window_log_row",
]
