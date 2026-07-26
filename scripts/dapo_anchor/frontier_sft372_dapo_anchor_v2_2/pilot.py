"""Run one isolated W0 V2.2 GPU window."""

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
    parser = argparse.ArgumentParser(prog="dapo-anchor-v2.2-pilot")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    from ksllm4rec_dapo_anchor_v2_2.config import validate_config
    from ksllm4rec_dapo_anchor_v2_2._infra.dapo_fingerprint import runtime_signature
    from ksllm4rec_dapo_anchor_v2_2.trainer import run_one_window_pilot

    validate_config(config)
    signature = runtime_signature(config, config_path=args.config)
    result = run_one_window_pilot(
        config,
        signature,
        output_dir=args.output_dir,
        device=args.device,
        aligned_cache=True,
    )
    print(json.dumps(result, ensure_ascii=False, separators=(",", ":")), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
