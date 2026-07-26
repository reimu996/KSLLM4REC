"""Resolve the V2.2 anchor weight from 16 unchanged-policy GPU windows."""

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
    parser = argparse.ArgumentParser(prog="dapo-anchor-v2.2-calibrate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    from ksllm4rec_dapo_anchor_v2_2.config import validate_config
    from ksllm4rec_dapo_anchor_v2_2._infra.dapo_fingerprint import runtime_signature
    from ksllm4rec_dapo_anchor_v2_2.trainer import run_calibration

    validate_config(config)
    expected = Path(config["anchor"]["calibration_path"]).resolve()
    if args.output.resolve() != expected:
        raise ValueError(f"Calibration output must be {expected}.")
    signature = runtime_signature(
        config,
        config_path=args.config,
        include_anchor_calibration=False,
    )
    report = run_calibration(
        config,
        signature,
        output_path=args.output,
        device=args.device,
        aligned_cache=True,
    )
    print(
        json.dumps(
            {
                "event": "anchor_calibration_complete",
                "lambda_calibrated": report["lambda_calibrated"],
                "valid_windows": report["valid_windows"],
                "optimizer_updates": report["optimizer_updates"],
                "output": str(args.output.resolve()),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
