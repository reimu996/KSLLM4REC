"""Run one isolated Multitask V1.1 optimization-window GPU pilot."""

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
    parser = argparse.ArgumentParser(prog="dapo-anchor-multitask-v1.1-e2-pilot")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    from ksllm4rec_dapo_anchor_multitask_v1_1.config import validate_config
    from ksllm4rec_dapo_anchor_multitask_v1_1._infra.dapo_fingerprint import (
        runtime_signature,
    )
    from ksllm4rec_dapo_anchor_multitask_v1_1.trainer import run_one_window_pilot

    validate_config(config)
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite pilot directory: {output}")
    signature = runtime_signature(config, config_path=args.config)
    report = run_one_window_pilot(
        config,
        signature,
        output_dir=output,
        device=args.device,
        aligned_cache=True,
    )
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
