"""Verify V2.2 multitask logs, recovery counters, and runtime identity."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[3]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(prog="verify-dapo-anchor-v2.2")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--expected-windows", type=int, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    from ksllm4rec_dapo_anchor_v2_2_multitask.config import validate_config
    from ksllm4rec_dapo_anchor_v2_2_multitask._infra.dapo_checkpoint import (
        validate_recovery_checkpoint,
    )
    from ksllm4rec_dapo_anchor_v2_2_multitask._infra.dapo_fingerprint import (
        runtime_signature,
    )
    from ksllm4rec_dapo_anchor_v2_2_multitask.verify import (
        validate_group_log_row,
        validate_run_summary,
        validate_window_log_row,
    )

    validate_config(config)
    signature = runtime_signature(config, config_path=args.config)
    run_dir = args.run_dir.resolve()
    if _json(run_dir / "runtime_signature.json") != signature:
        raise RuntimeError("Run runtime_signature.json differs from current frozen inputs.")
    if _json(run_dir / "resolved_config.json") != config:
        raise RuntimeError("Run resolved_config.json differs from the V2.2 config.")

    windows = [
        json.loads(line)
        for line in (run_dir / "windows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(windows) != args.expected_windows:
        raise RuntimeError(
            f"Expected {args.expected_windows} windows, found {len(windows)}."
        )
    for index, row in enumerate(windows):
        validate_window_log_row(row)
        if int(row["window_index"]) != index:
            raise RuntimeError("Window indices are not contiguous from zero.")

    groups = [
        json.loads(line)
        for line in (run_dir / "groups.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len(groups) != args.expected_windows * 32:
        raise RuntimeError("Group log does not contain exactly 32 rows per window.")
    for row in groups:
        validate_group_log_row(row)

    summary = _json(run_dir / "run_summary.json")
    validate_run_summary(summary)
    if int(summary["completed_windows"]) != args.expected_windows:
        raise RuntimeError("Run summary completed_windows differs from logs.")
    if args.require_complete and not bool(summary["complete"]):
        raise RuntimeError("The run is not marked complete.")

    latest = _json(run_dir / "recovery/latest.json")
    checkpoint = run_dir / "recovery" / latest["checkpoint"]
    cursor, state = validate_recovery_checkpoint(checkpoint, signature)
    if cursor.next_window_index != args.expected_windows:
        raise RuntimeError("Latest recovery does not cover all verified windows.")
    adam_steps = {
        int(value["step"].item())
        for value in state["optimizer"]["state"].values()
        if "step" in value
    }
    if adam_steps != {cursor.optimizer_update_step}:
        raise RuntimeError(
            f"AdamW state steps {sorted(adam_steps)} differ from cursor {cursor.optimizer_update_step}."
        )

    report = {
        "passed": True,
        "run_dir": str(run_dir),
        "windows": len(windows),
        "groups": len(groups),
        "anchor_max_weight": config["anchor"]["max_weight"],
        "rl_optimizer_update_step": cursor.rl_optimizer_update_step,
        "anchor_optimizer_update_step": cursor.anchor_optimizer_update_step,
        "optimizer_update_step": cursor.optimizer_update_step,
        "max_anchor_candidate_groups": max(
            (int(row["anchor_candidate_groups"]) for row in windows), default=0
        ),
        "max_anchor_to_rl_grad_ratio": max(
            (float(row["anchor_to_rl_grad_ratio"]) for row in windows), default=0.0
        ),
        "max_peak_reserved_gib": max(
            (float(row["peak_reserved_gib"]) for row in windows), default=0.0
        ),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
