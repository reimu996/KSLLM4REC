"""Fresh-process worker used only by the split-resume GPU gate."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

from . import contract
from .config import load_config
from .device import require_gpu_identity
from .fingerprint import runtime_signature
from .trainer import run_training


_STAGING_RECOVERY = re.compile(
    r"\.gates\.staging\.\d{8}T\d{6}Z\.\d+/pilot-resume-check"
)


def _approved_output(config: dict, output: Path) -> Path:
    profile = contract.frozen_profile(str(config["profile"]))
    actual = Path(output).expanduser().resolve()
    if profile.name != contract.SFT372_PROFILE:
        return actual
    fixed = profile.pilot_dir.with_name(f"{profile.pilot_dir.name}-resume-check")
    if actual == fixed.resolve():
        return actual
    try:
        relative = actual.relative_to(profile.log_dir.resolve()).as_posix()
    except ValueError as exc:
        raise RuntimeError(
            "Recovery gate output is outside the frozen log root."
        ) from exc
    if _STAGING_RECOVERY.fullmatch(relative) is None:
        raise RuntimeError("Recovery gate output is not an approved staging path.")
    return actual


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="ksllm4rec-rloo-dapo-recovery-worker")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-signature", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=contract.EXECUTION_DEVICE)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    profile = contract.frozen_profile(str(config["profile"]))
    if (
        profile.name == contract.SFT372_PROFILE
        and args.config.expanduser().resolve() != profile.config_path.resolve()
    ):
        raise RuntimeError("Recovery worker config is not the frozen profile config.")
    if profile.name == contract.SFT372_PROFILE and (
        args.device != contract.EXECUTION_DEVICE
    ):
        raise RuntimeError("Recovery worker device differs from the frozen device.")
    if profile.name == contract.SFT372_PROFILE and os.environ.get(
        "CUDA_VISIBLE_DEVICES"
    ) != (contract.EXECUTION_CUDA_VISIBLE_DEVICES):
        raise RuntimeError("Recovery worker CUDA visibility differs from the contract.")
    if profile.name == contract.SFT372_PROFILE:
        require_gpu_identity(contract.EXPECTED_GPU_IDENTITY, args.device)
    signature = runtime_signature(config, config_path=args.config)
    if signature["sha256"] != args.expected_signature:
        raise RuntimeError("Recovery worker runtime signature changed.")
    output = _approved_output(config, args.output_dir)
    summary = run_training(
        config,
        signature,
        output_dir=output,
        device=args.device,
        resume=args.resume,
        stop_after_windows=1,
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
