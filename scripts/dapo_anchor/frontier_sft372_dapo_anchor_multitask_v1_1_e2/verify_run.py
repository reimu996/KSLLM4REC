"""Verify the complete V1.1 two-epoch formal run."""

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


def _json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(prog="verify-dapo-anchor-multitask-v1_1")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    from ksllm4rec_dapo_anchor_multitask_v1_1.config import validate_config
    from ksllm4rec_dapo_anchor_multitask_v1_1._infra.dapo_fingerprint import (
        runtime_signature,
    )
    from ksllm4rec_dapo_anchor_multitask_v1_1.checkpoint import (
        validate_recovery_checkpoint,
    )
    from ksllm4rec_dapo_anchor_multitask_v1_1.verify import (
        validate_run_directory_contents,
        verify_run_directory,
    )

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    run_dir = args.run_dir.resolve()
    if run_dir != Path(config["output"]["run_dir"]).resolve():
        raise ValueError("Run directory differs from the frozen config.")
    validate_run_directory_contents(run_dir)
    signature = runtime_signature(config, config_path=args.config)
    if _json(run_dir / "resolved_config.json") != config:
        raise RuntimeError("Run config differs from the frozen config.")
    if _json(run_dir / "runtime_signature.json") != signature:
        raise RuntimeError("Run signature differs from current frozen inputs.")
    summary = _json(run_dir / "run_summary.json")
    complete = summary.get("complete") is True
    if args.require_complete and not complete:
        raise RuntimeError("Run is not complete.")
    if not complete:
        raise RuntimeError(
            "Partial runs are recovered from their atomic checkpoint; complete "
            "coverage verification requires both source epochs."
        )

    result = verify_run_directory(run_dir, config)
    latest = _json(run_dir / "recovery/latest.json")
    checkpoint = run_dir / "recovery" / str(latest["checkpoint"])
    recovery, _training_state = validate_recovery_checkpoint(checkpoint, signature)
    if int(recovery.next_source_block) != int(summary["completed_source_blocks"]):
        raise RuntimeError("Latest recovery source cursor differs from summary.")
    if recovery.source_plan_sha256 != summary["source_plan_sha256"]:
        raise RuntimeError("Latest recovery source plan differs from summary.")
    if list(recovery.epoch_plan_sha256s) != summary["epoch_plan_sha256s"]:
        raise RuntimeError("Latest recovery epoch plan hashes differ from summary.")
    if int(recovery.total_optimizer_steps) != int(summary["total_optimizer_steps"]):
        raise RuntimeError("Latest recovery optimizer cursor differs from summary.")
    if int(recovery.anchor_log_rows) != int(summary["anchor_groups"]):
        raise RuntimeError("Latest recovery Anchor cursor differs from summary.")
    report = {
        "passed": True,
        "complete": True,
        "source_groups": result.source.source_groups,
        "candidate_rollouts": result.source.candidate_rollouts,
        "optimization_windows": result.windows.optimization_windows,
        "completed_policy_steps": result.windows.policy_optimizer_steps,
        "anchor_groups": result.anchor_groups,
        "latest_recovery": latest["checkpoint"],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
