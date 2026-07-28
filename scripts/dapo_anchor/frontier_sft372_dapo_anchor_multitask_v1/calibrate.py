"""Measure the fixed anchor coefficient without updating the policy."""

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
    parser = argparse.ArgumentParser(prog="dapo-anchor-multitask-v1-calibrate")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    from ksllm4rec_dapo_anchor_multitask_v1.config import validate_config
    from ksllm4rec_dapo_anchor_multitask_v1._infra.dapo_fingerprint import (
        runtime_signature,
    )
    from ksllm4rec_dapo_anchor_multitask_v1.trainer import run_calibration

    validate_config(config)
    expected = Path(config["anchor"]["calibration_path"]).resolve()
    if args.output.resolve() != expected:
        raise ValueError(f"Calibration output must be {expected}.")
    if expected.exists():
        raise FileExistsError(f"Refusing to overwrite calibration: {expected}")
    signature = runtime_signature(
        config,
        config_path=args.config,
        include_anchor_calibration=False,
    )
    report = run_calibration(
        config,
        signature,
        output_path=expected,
        device=args.device,
        aligned_cache=True,
    )
    print(
        json.dumps(
            {
                "event": "anchor_calibration_complete",
                "source_blocks": report["source_blocks"],
                "policy_optimizer_steps": report["policy_optimizer_steps"],
                "model_unchanged": report["model_unchanged"],
                "lambda_calibrated": report["lambda_calibrated"],
                "output": str(expected),
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

