"""One fresh-process segment of the offline V1.1 recovery audit."""

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


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="dapo-anchor-multitask-v1.1-e2-recovery-worker"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-signature", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--stop-after-source-blocks", type=int, required=True)
    parser.add_argument("--inject-fault-after-source-blocks", type=int)
    args = parser.parse_args()

    from ksllm4rec_dapo_anchor_multitask_v1_1.config import validate_config
    from ksllm4rec_dapo_anchor_multitask_v1_1._infra.dapo_fingerprint import (
        runtime_signature,
    )
    from ksllm4rec_dapo_anchor_multitask_v1_1.trainer import (
        RecoveryAuditInjectedFailure,
        run_training,
    )

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    signature = runtime_signature(config, config_path=args.config)
    if signature["sha256"] != args.expected_signature:
        raise RuntimeError("Recovery worker runtime signature changed.")
    try:
        summary = run_training(
            config,
            signature,
            output_dir=args.output_dir.resolve(),
            device=args.device,
            resume=args.resume,
            stop_after_source_blocks=args.stop_after_source_blocks,
            aligned_cache=True,
            audit_mode=True,
            inject_fault_after_source_blocks=args.inject_fault_after_source_blocks,
        )
    except RecoveryAuditInjectedFailure as exc:
        print(json.dumps({"event": "injected_fault", "message": str(exc)}, ensure_ascii=False))
        return 42
    print(json.dumps({"event": "completed", "summary": summary}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
