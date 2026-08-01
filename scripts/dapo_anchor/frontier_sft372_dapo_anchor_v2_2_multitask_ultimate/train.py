"""DAPO-Anchor V2.2 training launcher.

Reads yaml -> builds runtime_signature via
ksllm4rec_dapo_anchor_v2_2_multitask_ultimate._infra.dapo_fingerprint.runtime_signature
and invokes ksllm4rec_dapo_anchor_v2_2_multitask_ultimate.trainer.run_training.

The trainer itself writes every training artefact (windows.jsonl,
groups.jsonl, recovery/, epoch-XX-adapter/, resolved_config.json,
runtime_signature.json, run_summary.json). This launcher only wraps that
call with a JSON start/end marker on stdout so status.sh can extract the
invocation exit code from the tee'd log.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import yaml

# Make sure the project's src/ is importable even when this launcher is
# invoked without PYTHONPATH set (common.sh already sets it; direct python
# invocation might not).
_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[3]
_SRC = _PROJECT_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> int:
    parser = argparse.ArgumentParser(prog="dapo-anchor-train")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument(
        "--stop-after-windows",
        type=int,
        default=None,
        help="If set, exit cleanly after this many windows (for smoke).",
    )
    parser.add_argument(
        "--dense-scoring",
        action="store_true",
        help="Run in dense scoring mode (audit only, disables kv-cache fast path).",
    )
    args = parser.parse_args()

    if not args.config.is_file():
        print(f"Config not found: {args.config}", file=sys.stderr)
        return 2

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    from ksllm4rec_dapo_anchor_v2_2_multitask_ultimate.config import validate_config

    validate_config(config)

    # Build runtime signature (all frozen inputs + code + software versions)
    from ksllm4rec_dapo_anchor_v2_2_multitask_ultimate._infra.dapo_fingerprint import (
        runtime_signature,
    )

    signature = runtime_signature(config, config_path=args.config)

    # Emit start marker (parsed by status.sh)
    start_event = {
        "event": "train_invocation_start",
        "unix_time": int(time.time()),
        "pid": os.getpid(),
        "output_dir": str(args.output_dir.resolve()),
        "config": str(args.config.resolve()),
        "signature_sha256": signature["sha256"],
        "device": args.device,
        "resume": not args.no_resume,
        "stop_after_windows": args.stop_after_windows,
        "dense_scoring": bool(args.dense_scoring),
        "formal": bool(args.formal),
    }
    print(json.dumps(start_event, ensure_ascii=False), flush=True)

    # Kick off training
    from ksllm4rec_dapo_anchor_v2_2_multitask_ultimate.trainer import run_training

    try:
        result = run_training(
            config,
            signature,
            output_dir=args.output_dir,
            device=args.device,
            resume=not args.no_resume,
            stop_after_windows=args.stop_after_windows,
            aligned_cache=True,
            dense_scoring=bool(args.dense_scoring),
            formal=bool(args.formal),
        )
        end_event = {
            "event": "train_invocation_end",
            "unix_time": int(time.time()),
            "exit_code": 0,
            "summary": result,
        }
        print(json.dumps(end_event, ensure_ascii=False), flush=True)
        return 0
    except BaseException as exc:  # noqa: BLE001 - want KeyboardInterrupt too
        import traceback

        traceback_text = traceback.format_exc()
        end_event = {
            "event": "train_invocation_end",
            "unix_time": int(time.time()),
            "exit_code": 1,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
        }
        print(json.dumps(end_event, ensure_ascii=False), flush=True)
        print(traceback_text, file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
