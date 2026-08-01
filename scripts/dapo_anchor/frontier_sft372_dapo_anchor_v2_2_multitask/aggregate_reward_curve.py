#!/usr/bin/env python3
"""Aggregate the training-time online reward/exact curve for the multitask run.

Spec: "训练在线 reward/exact 曲线还原（基于已落盘的 groups.jsonl）" FR-001..FR-005.
Read-only aggregation of:
  artifacts/rloo/runs/dapo_anchor_v2_2_multitask_sft372_e2/groups.jsonl
  artifacts/rloo/runs/dapo_anchor_v2_2_multitask_sft372_e2/windows.jsonl
Outputs (same run dir):
  reward_curve.jsonl            one row per window
  reward_curve_summary.csv      condensed per-window table
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

RUN_DIR = Path(
    "/home/lyc/REC_PROJECTS/KSLLM4REC/artifacts/rloo/runs/"
    "dapo_anchor_v2_2_multitask_sft372_e2"
)
GROUPS = RUN_DIR / "groups.jsonl"
WINDOWS = RUN_DIR / "windows.jsonl"
OUT_JSONL = RUN_DIR / "reward_curve.jsonl"
OUT_CSV = RUN_DIR / "reward_curve_summary.csv"

GROUP_SIZE = 16
GROUPS_PER_WINDOW = 32
TIERS = ("exact", "same_ab", "same_a", "same_domain", "other_domain")
TIER_VALUES = {"exact": 1.0, "same_ab": 0.4, "same_a": 0.15,
               "same_domain": 0.01, "other_domain": 0.0}


def load_windows() -> dict[int, dict]:
    """windows.jsonl: window_index -> row."""
    out = {}
    for line in open(WINDOWS, encoding="utf-8"):
        row = json.loads(line)
        out[int(row["window_index"])] = row
    return out


def load_groups() -> dict[int, list[dict]]:
    """groups.jsonl: window_index -> list of group rows (32 each)."""
    per_window: dict[int, list[dict]] = {}
    for line in open(GROUPS, encoding="utf-8"):
        row = json.loads(line)
        per_window.setdefault(int(row["window_index"]), []).append(row)
    return per_window


def window_stats(window_index: int, groups: list[dict], win: dict) -> dict:
    rewards = [value for g in groups for value in g["rewards"]]
    tiers = Counter(t for g in groups for t in g["reward_tiers"])
    exact_candidates = sum(1 for v in rewards if v == TIER_VALUES["exact"])
    exact_groups = sum(
        1 for g in groups if TIER_VALUES["exact"] in g["rewards"]
    )
    filtered = int(win["filtered_group_count"])
    all_groups = GROUPS_PER_WINDOW + filtered
    return {
        "window_index": window_index,
        "effective_epoch": int(win["effective_epoch"]),
        "reward_mean": round(statistics.mean(rewards), 6),
        "reward_std": round(statistics.pstdev(rewards), 6),
        "exact_candidate_count": exact_candidates,
        "exact_candidate_rate": round(exact_candidates / len(rewards), 6),
        "exact_group_count": exact_groups,
        "exact_group_rate_effective": round(exact_groups / GROUPS_PER_WINDOW, 6),
        "exact_group_rate_all": round(exact_groups / all_groups, 6),
        "tier_exact": tiers["exact"],
        "tier_same_ab": tiers["same_ab"],
        "tier_same_a": tiers["same_a"],
        "tier_same_domain": tiers["same_domain"],
        "tier_other": tiers["other_domain"],
        "task_recommendation": sum(1 for g in groups if g["task"] == "recommendation"),
        "task_text_to_sid": sum(1 for g in groups if g["task"] == "item_text_to_sid"),
        "filtered_group_count": filtered,
        "anchor_candidate_groups": int(win["anchor_candidate_groups"]),
    }


def cross_check(groups_by_window: dict[int, list[dict]], windows: dict[int, dict]) -> None:
    """Spec FR-003: group_ids set equality per window."""
    for w, win in windows.items():
        if w not in groups_by_window:
            raise SystemExit(f"window {w} missing in groups.jsonl")
        ids = {g["group_key"] for g in groups_by_window[w]}
        recorded = set(win["group_ids"])
        if ids != recorded:
            raise SystemExit(
                f"window {w}: group_ids mismatch "
                f"(groups {len(ids)} vs windows {len(recorded)})"
            )
        for g in groups_by_window[w]:
            if int(g["anchor_candidate_groups"]) != int(win["anchor_candidate_groups"]):
                raise SystemExit(
                    f"window {w}: anchor_candidate_groups mismatch"
                )


def main() -> int:
    if OUT_JSONL.exists() or OUT_CSV.exists():
        raise SystemExit("Outputs exist; delete them first to regenerate.")
    windows = load_windows()
    groups_by_window = load_groups()
    if len(windows) != 642 or len(groups_by_window) != 642:
        raise SystemExit(
            f"expected 642 windows, got windows={len(windows)} groups={len(groups_by_window)}"
        )
    for w, gs in groups_by_window.items():
        if len(gs) != GROUPS_PER_WINDOW:
            raise SystemExit(f"window {w}: {len(gs)} groups, expected 32")
    cross_check(groups_by_window, windows)

    rows = [window_stats(w, groups_by_window[w], windows[w])
            for w in sorted(windows)]
    with open(OUT_JSONL, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    header = list(rows[0].keys())
    with open(OUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow([row[k] for k in header])

    first, last = rows[0], rows[-1]
    print(f"OK: {len(rows)} windows -> {OUT_JSONL.name}, {OUT_CSV.name}")
    print(f"w{first['window_index']}: reward_mean={first['reward_mean']} "
          f"exact_cand={first['exact_candidate_count']} "
          f"exact_grp={first['exact_group_count']}/32")
    print(f"w{last['window_index']}: reward_mean={last['reward_mean']} "
          f"exact_cand={last['exact_candidate_count']} "
          f"exact_grp={last['exact_group_count']}/32")
    return 0


if __name__ == "__main__":
    sys.exit(main())
