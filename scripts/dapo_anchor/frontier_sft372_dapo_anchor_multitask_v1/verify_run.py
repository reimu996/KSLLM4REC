"""Verify immutable identity and one-pass source accounting for a V1 run."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import yaml


_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[3]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL {path}:{line_number}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"Expected object at {path}:{line_number}")
            yield value


def main() -> int:
    parser = argparse.ArgumentParser(prog="verify-dapo-anchor-multitask-v1")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    from ksllm4rec_dapo_anchor_multitask_v1.config import validate_config
    from ksllm4rec_dapo_anchor_multitask_v1._infra.dapo_fingerprint import (
        runtime_signature,
    )
    from ksllm4rec_dapo_anchor_multitask_v1.checkpoint import (
        validate_recovery_checkpoint,
    )
    from ksllm4rec_dapo_anchor_multitask_v1.source_blocks import (
        build_source_blocks,
    )
    from ksllm4rec_dapo_anchor_multitask_v1.verify import (
        validate_run_directory_contents,
        verify_run_directory,
    )

    validate_config(config)
    signature = runtime_signature(config, config_path=args.config)
    run_dir = args.run_dir.resolve()
    if run_dir != Path(config["output"]["run_dir"]).resolve():
        raise ValueError("Run directory differs from the frozen config.")
    validate_run_directory_contents(run_dir)
    if _json(run_dir / "resolved_config.json") != config:
        raise RuntimeError("Run config differs from the frozen config.")
    if _json(run_dir / "runtime_signature.json") != signature:
        raise RuntimeError("Run signature differs from current frozen inputs.")
    verify_run_directory(run_dir, config)

    summary = _json(run_dir / "run_summary.json")
    complete = summary.get("complete") is True
    if args.require_complete and not complete:
        raise RuntimeError("Run is not complete.")
    source_rows = list(_jsonl(run_dir / "source_groups.jsonl"))
    if int(summary["source_groups"]) != len(source_rows):
        raise RuntimeError("Summary source-group count differs from source audit.")
    expected_plan: list[tuple[int, str, int]] = []
    for block in build_source_blocks(
        recommendation_groups=int(config["data"]["recommendation_groups"]),
        text_to_sid_groups=int(config["data"]["text_to_sid_groups"]),
        blocks=int(config["data"]["source_blocks"]),
    ):
        for source_index in range(
            block.recommendation.start, block.recommendation.stop
        ):
            expected_plan.append((block.index, "recommendation", source_index))
        if config["experiments"]["active"]["text_to_sid_enabled"]:
            for source_index in range(
                block.text_to_sid.start, block.text_to_sid.stop
            ):
                expected_plan.append((block.index, "item_text_to_sid", source_index))
    observed_plan = [
        (int(row["block_index"]), str(row["task"]), int(row["source_index"]))
        for row in source_rows
    ]
    if len(observed_plan) > len(expected_plan):
        raise RuntimeError("Source audit contains more rows than the fixed source plan.")
    expected_prefix = expected_plan[: len(observed_plan)]
    if observed_plan != expected_prefix:
        mismatch = next(
            (index, expected, observed)
            for index, (expected, observed) in enumerate(
                zip(expected_prefix, observed_plan, strict=True)
            )
            if expected != observed
        )
        raise RuntimeError(f"Source audit differs from the floor-partition plan: {mismatch!r}")
    if any(int(row["rollout_count"]) != 16 for row in source_rows):
        raise RuntimeError("Every source group must have exactly 16 candidates.")

    windows = list(_jsonl(run_dir / "windows.jsonl"))
    all_policy_ids: list[tuple[str, str]] = []
    for expected_index, row in enumerate(windows):
        if int(row["optimization_window_index"]) != expected_index:
            raise RuntimeError("Optimization-window indices are not contiguous.")
        if int(row["anchor_optimizer_steps"]) != 0 or int(row["k"]) != 1:
            raise RuntimeError("Anchor must be merged and rollout reuse K must equal one.")
        ratio = float(row["anchor_to_rl_gradient_ratio"])
        if not math.isfinite(ratio) or ratio > 0.1 + 1.0e-12:
            raise RuntimeError("Anchor gradient exceeds the 10% budget.")
        ids = [
            (str(task), str(group_id))
            for step in row["steps"]
            for task, group_id in zip(
                step["tasks"], step["group_ids"], strict=True
            )
        ]
        if len(ids) != len(set(ids)):
            raise RuntimeError("A rollout group was reused inside one optimization window.")
        if any(not 1 <= len(step["group_ids"]) <= 8 for step in row["steps"]):
            raise RuntimeError("A policy minibatch must contain one to eight groups.")
        if len(ids) != int(row["rl_groups"]):
            raise RuntimeError("Window RL-group count differs from its policy steps.")
        all_policy_ids.extend(ids)

    group_rows = list(_jsonl(run_dir / "groups.jsonl"))
    logged_policy_ids = [
        (str(row["task"]), str(row["group_id"])) for row in group_rows
    ]
    if sorted(logged_policy_ids) != sorted(all_policy_ids):
        raise RuntimeError("Group audit rows differ from policy-step group identities.")
    if len(logged_policy_ids) != len(set(logged_policy_ids)):
        raise RuntimeError("A source group entered RL more than once.")
    if any(int(row["gt_injection_count"]) != 0 or int(row["k"]) != 1 for row in group_rows):
        raise RuntimeError("RL group audit violates no-injection or K=1.")

    if complete:
        expected_groups = int(config["data"]["recommendation_groups"])
        if config["experiments"]["active"]["text_to_sid_enabled"]:
            expected_groups += int(config["data"]["text_to_sid_groups"])
        if len(source_rows) != expected_groups:
            raise RuntimeError(
                f"Complete run must contain {expected_groups} source groups."
            )
        if int(summary["candidate_rollouts"]) != expected_groups * 16:
            raise RuntimeError("Complete run candidate total is not exact.")
        if int(summary["completed_source_blocks"]) != int(config["data"]["source_blocks"]):
            raise RuntimeError("Complete run did not consume all source blocks.")
        if observed_plan != expected_plan:
            raise RuntimeError("Complete run source audit is not the exact source plan.")

    if int(summary["optimization_windows"]) != len(windows):
        raise RuntimeError("Summary optimization-window count differs from logs.")
    if int(summary["rl_groups"]) != len(group_rows):
        raise RuntimeError("Summary RL-group count differs from logs.")
    policy_steps = sum(int(row["policy_optimizer_steps"]) for row in windows)
    if int(summary["completed_policy_steps"]) != policy_steps:
        raise RuntimeError("Summary policy-step count differs from logs.")
    if int(summary["total_optimizer_steps"]) != (
        policy_steps + int(summary["final_auxiliary_flush_steps"])
    ):
        raise RuntimeError("Summary total optimizer-step count is inconsistent.")

    latest = _json(run_dir / "recovery/latest.json")
    checkpoint = run_dir / "recovery" / str(latest["checkpoint"])
    recovery, _training_state = validate_recovery_checkpoint(checkpoint, signature)
    if int(recovery.next_source_block) != int(summary["completed_source_blocks"]):
        raise RuntimeError("Latest recovery source cursor differs from summary.")
    if int(recovery.total_optimizer_steps) != int(summary["total_optimizer_steps"]):
        raise RuntimeError("Latest recovery optimizer cursor differs from summary.")
    if recovery.final_auxiliary_flush != summary["final_auxiliary_flush"]:
        raise RuntimeError("Latest recovery final auxiliary audit differs from summary.")
    report = {
        "passed": True,
        "complete": complete,
        "source_groups": len(source_rows),
        "candidate_rollouts": int(summary["candidate_rollouts"]),
        "optimization_windows": len(windows),
        "completed_policy_steps": int(summary["completed_policy_steps"]),
        "final_auxiliary_flush_steps": int(summary["final_auxiliary_flush_steps"]),
        "latest_recovery": latest["checkpoint"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
