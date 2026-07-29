"""Frozen formal launcher for DAPO-Anchor-Multitask V1.1 two-epoch training."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import yaml


_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[3]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def main() -> int:
    parser = argparse.ArgumentParser(prog="dapo-anchor-multitask-v1.1-e2-train")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--formal", action="store_true")
    parser.add_argument(
        "--stop-after-source-blocks",
        type=int,
        default=None,
        help="Cleanly stop after N additional source blocks; used only for recovery audits.",
    )
    parser.add_argument("--dense-scoring", action="store_true")
    args = parser.parse_args()

    if args.formal and args.stop_after_source_blocks is not None:
        raise ValueError("Formal training cannot use the recovery-audit stop hook.")
    if args.formal and args.dense_scoring:
        raise ValueError("Formal training cannot use the dense-scoring test hook.")

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    from ksllm4rec_dapo_anchor_multitask_v1_1.config import validate_config
    from ksllm4rec_dapo_anchor_multitask_v1_1._infra.dapo_fingerprint import (
        runtime_signature,
    )

    validate_config(config)
    output = args.output_dir.resolve()
    expected = Path(config["output"]["run_dir"]).resolve()
    if output != expected:
        raise ValueError(f"Formal output must be {expected}.")
    if output.exists() and any(output.iterdir()):
        required = {
            output / "resolved_config.json",
            output / "runtime_signature.json",
            output / "recovery/latest.json",
        }
        missing = sorted(str(path) for path in required if not path.is_file())
        if missing:
            raise FileExistsError(
                "Refusing to overwrite a non-recoverable run directory; missing "
                + ", ".join(missing)
            )

    signature = runtime_signature(config, config_path=args.config)
    event = {
        "event": "train_invocation_start",
        "unix_time": int(time.time()),
        "pid": os.getpid(),
        "config": str(args.config.resolve()),
        "output_dir": str(output),
        "signature_sha256": signature["sha256"],
        "device": args.device,
        "resume": output.exists() and any(output.iterdir()),
        "stop_after_source_blocks": args.stop_after_source_blocks,
        "dense_scoring": args.dense_scoring,
        "formal": args.formal,
    }
    print(json.dumps(event, ensure_ascii=False, separators=(",", ":")), flush=True)

    from ksllm4rec_dapo_anchor_multitask_v1_1.trainer import run_training

    try:
        result = run_training(
            config,
            signature,
            output_dir=output,
            device=args.device,
            resume=True,
            stop_after_source_blocks=args.stop_after_source_blocks,
            aligned_cache=True,
            dense_scoring=args.dense_scoring,
            formal=args.formal,
        )
    except BaseException as exc:  # keep a machine-readable failure marker
        print(
            json.dumps(
                {
                    "event": "train_invocation_end",
                    "unix_time": int(time.time()),
                    "exit_code": 1,
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                },
                ensure_ascii=False,
                separators=(",", ":"),
            ),
            flush=True,
        )
        traceback.print_exc()
        return 1
    print(
        json.dumps(
            {
                "event": "train_invocation_end",
                "unix_time": int(time.time()),
                "exit_code": 0,
                "summary": result,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
