"""Run continuous and interrupted fresh-process V1.1 audit trajectories."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

import yaml


_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[3]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _run(command: list[str], *, expected: int) -> None:
    completed = subprocess.run(command, check=False)
    if completed.returncode != expected:
        raise RuntimeError(
            f"Recovery audit worker returned {completed.returncode}, expected {expected}."
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="run-dapo-anchor-multitask-v1.1-e2-recovery-audit"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--source-blocks", type=int, default=4)
    parser.add_argument("--checkpoint-blocks", type=int, default=2)
    parser.add_argument("--audit-id")
    args = parser.parse_args()
    if args.source_blocks < 2:
        raise ValueError("Recovery audit needs at least two source blocks.")
    if not 1 <= args.checkpoint_blocks < args.source_blocks:
        raise ValueError("checkpoint-blocks must be within the audited source range.")

    from ksllm4rec_dapo_anchor_multitask_v1_1 import contract
    from ksllm4rec_dapo_anchor_multitask_v1_1.config import validate_config
    from ksllm4rec_dapo_anchor_multitask_v1_1._infra.dapo_fingerprint import (
        runtime_signature,
    )
    from ksllm4rec_dapo_anchor_multitask_v1_1.recovery_audit import (
        compare_recovery_trajectories,
    )

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    signature = runtime_signature(config, config_path=args.config)
    audit_id = args.audit_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = (contract.RECOVERY_AUDIT_ROOT / audit_id).resolve()
    if root.exists():
        raise FileExistsError(root)
    continuous = root / "continuous"
    recovered = root / "interrupted"
    worker = _HERE.with_name("recovery_audit_worker.py")
    base = [
        sys.executable,
        str(worker),
        "--config",
        str(args.config.resolve()),
        "--expected-signature",
        signature["sha256"],
        "--device",
        args.device,
    ]
    _run(
        [
            *base,
            "--output-dir",
            str(continuous),
            "--stop-after-source-blocks",
            str(args.source_blocks),
        ],
        expected=0,
    )
    _run(
        [
            *base,
            "--output-dir",
            str(recovered),
            "--stop-after-source-blocks",
            str(args.checkpoint_blocks),
        ],
        expected=0,
    )
    remaining = args.source_blocks - args.checkpoint_blocks
    _run(
        [
            *base,
            "--output-dir",
            str(recovered),
            "--resume",
            "--stop-after-source-blocks",
            str(remaining),
            "--inject-fault-after-source-blocks",
            str(args.source_blocks),
        ],
        expected=42,
    )
    _run(
        [
            *base,
            "--output-dir",
            str(recovered),
            "--resume",
            "--stop-after-source-blocks",
            str(remaining),
        ],
        expected=0,
    )
    report = compare_recovery_trajectories(continuous, recovered, signature)
    report.update(
        {
            "audit_id": audit_id,
            "source_blocks": args.source_blocks,
            "checkpoint_blocks": args.checkpoint_blocks,
            "signature_sha256": signature["sha256"],
        }
    )
    if not root.is_dir():
        raise RuntimeError("Recovery audit workers did not create their audit root.")
    (root / "recovery_audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
