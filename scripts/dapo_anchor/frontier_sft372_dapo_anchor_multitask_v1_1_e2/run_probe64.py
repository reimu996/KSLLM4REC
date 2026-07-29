"""Run the fixed offline Probe64 against one explicit policy adapter."""

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
    parser = argparse.ArgumentParser(prog="dapo-anchor-multitask-v1.1-e2-probe64")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--index",
        type=int,
        action="append",
        default=None,
        help="Evaluate one sorted probe index; repeat for a partial GPU pilot.",
    )
    args = parser.parse_args()

    from ksllm4rec_dapo_anchor_multitask_v1_1._infra.dapo_fingerprint import (
        runtime_signature,
    )
    from ksllm4rec_dapo_anchor_multitask_v1_1.config import validate_config
    from ksllm4rec_dapo_anchor_multitask_v1_1.evaluation import run_fixed_probe64

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    adapter = args.adapter.resolve()
    for name in ("adapter_config.json", "adapter_model.safetensors"):
        if not (adapter / name).is_file():
            raise FileNotFoundError(adapter / name)
    signature = runtime_signature(config, config_path=args.config)
    report = run_fixed_probe64(
        config,
        output_path=args.output.resolve(),
        device=args.device,
        policy_adapter_path=adapter,
        indices=args.index,
        runtime_signature_sha256=str(signature["sha256"]),
    )
    print(
        json.dumps(
            {
                "event": "probe64_complete",
                "complete": report["complete"],
                "result_count": report["result_count"],
                "pass_count": report["pass_count"],
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
