"""Deterministic gate verifiers for DAPO-Anchor training.

Difference from ksllm4rec_rloo_dapo.verify:
  - anchor_groups 允许 > 0.
  - 新增 anchor 相关指标的校验.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from . import contract


# ── Row-level checks ─────────────────────────────────────────────────

def validate_window_log_row(row: Mapping[str, Any]) -> None:
    """Basic sanity checks for a DAPO-Anchor window log row."""

    required_keys = {
        "window_index", "effective_groups", "filter_groups",
        "anchor_groups", "anchor_loss", "anchor_grad_norm",
        "optimizer_updates", "steps",
    }
    missing = required_keys - set(row)
    if missing:
        raise ValueError(f"Window log row missing keys: {missing}")

    window_index = int(row["window_index"])
    if window_index < 0:
        raise ValueError(f"Negative window index: {window_index}")

    effective = int(row.get("effective_groups", 0))
    if effective != contract.EFFECTIVE_GROUPS_PER_WINDOW:
        raise ValueError(
            f"Window {window_index}: expected {contract.EFFECTIVE_GROUPS_PER_WINDOW} "
            f"effective groups, got {effective}."
        )

    anchor_groups = int(row.get("anchor_groups", 0))
    if anchor_groups < 0 or anchor_groups > contract.EXPECTED_RECOMMEND_GROUPS:
        raise ValueError(
            f"Window {window_index}: anchor_groups {anchor_groups} "
            f"exceeds dataset size {contract.EXPECTED_RECOMMEND_GROUPS}."
        )

    anchor_loss = float(row.get("anchor_loss", 0.0))
    if not math.isfinite(anchor_loss):
        raise ValueError(
            f"Window {window_index}: anchor_loss is non-finite."
        )

    anchor_grad_norm = float(row.get("anchor_grad_norm", 0.0))
    if anchor_grad_norm < 0.0 or not math.isfinite(anchor_grad_norm):
        raise ValueError(
            f"Window {window_index}: anchor_grad_norm is negative or non-finite."
        )

    optimizer_updates = int(row.get("optimizer_updates", 0))
    if optimizer_updates != contract.OPTIMIZER_UPDATES_PER_WINDOW:
        raise ValueError(
            f"Window {window_index}: expected "
            f"{contract.OPTIMIZER_UPDATES_PER_WINDOW} optimizer updates, "
            f"got {optimizer_updates}."
        )


def validate_group_log_row(row: Mapping[str, Any]) -> None:
    """Basic sanity checks for a group log row."""

    required_keys = {
        "window_index", "group_id", "effective",
        "anchor_groups", "gt_injection_count",
    }
    missing = required_keys - set(row)
    if missing:
        raise ValueError(f"Group log row missing keys: {missing}")

    window_index = int(row["window_index"])
    anchor_groups = int(row.get("anchor_groups", 0))
    if anchor_groups < 0:
        raise ValueError(
            f"Window {window_index}: negative anchor_groups in group row."
        )

    rewards = row.get("rewards", [])
    if len(rewards) != contract.GROUP_SIZE:
        raise ValueError(
            f"Window {window_index}: expected {contract.GROUP_SIZE} rewards, "
            f"got {len(rewards)}."
        )
    if not all(math.isfinite(value) for value in rewards):
        raise ValueError(
            f"Window {window_index}: non-finite reward value."
        )


# ── Summary checks ───────────────────────────────────────────────────

def validate_run_summary(
    summary: Mapping[str, Any],
    *,
    expected_windows: int | None = None,
    allow_anchor: bool = True,
) -> None:
    """Full-run validation after completion.

    Args:
        summary: run_summary.json dict.
        expected_windows: expected total windows (None = skip check).
        allow_anchor: if True, anchor_groups can be > 0.
    """
    required = {
        "completed_windows", "optimizer_update_step", "complete",
    }
    missing = required - set(summary)
    if missing:
        raise ValueError(f"Run summary missing keys: {missing}")

    completed = int(summary["completed_windows"])
    if expected_windows is not None and completed != expected_windows:
        raise ValueError(
            f"Expected {expected_windows} completed windows, got {completed}."
        )

    if not allow_anchor:
        anchor_count = int(summary.get("anchor_groups", 0))
        if anchor_count != 0:
            raise ValueError(
                f"Expected anchor_groups=0, got {anchor_count}."
            )

    complete = bool(summary.get("complete", False))
    if expected_windows is not None and complete != (completed == expected_windows):
        raise ValueError("complete flag contradicts completed_windows count.")


# ── Convenience: JSONL iteration ─────────────────────────────────────

def check_window_log(
    window_log: Path,
) -> tuple[int, int, float]:
    """Aggregate anchor metrics from a completed window log.

    Returns (window_count, total_anchor_groups, mean_anchor_loss).
    """
    total_anchor_groups = 0
    total_anchor_loss = 0.0
    window_count = 0
    with window_log.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            validate_window_log_row(row)
            total_anchor_groups += int(row.get("anchor_groups", 0))
            total_anchor_loss += float(row.get("anchor_loss", 0.0))
            window_count += 1
    mean_loss = total_anchor_loss / max(window_count, 1)
    return window_count, total_anchor_groups, mean_loss


__all__ = [
    "check_window_log",
    "validate_group_log_row",
    "validate_run_summary",
    "validate_window_log_row",
]
