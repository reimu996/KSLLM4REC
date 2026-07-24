"""Command-line entry points for gates, pilot, formal training, and verification."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
from typing import Any

from . import contract
from .config import load_config
from .cpu_gate import run_cpu_gate, validate_cpu_gate_report
from .device import gpu_identity, require_gpu_identity
from .fingerprint import runtime_signature
from .gates import (
    load_and_validate_gate_reports,
    memory_gate,
    pilot_gate,
    probability_gate,
    structure_gate,
    throughput_gate,
)
from .trainer import run_training
from .verify import verify_run


_STAGING_DIRECTORY = re.compile(r"\.gates\.staging\.\d{8}T\d{6}Z\.\d+")


def _resolved(path: Path) -> Path:
    return Path(path).expanduser().resolve()


def _require_path(actual: Path, expected: Path, label: str) -> None:
    if _resolved(actual) != _resolved(expected):
        raise RuntimeError(
            f"{label} must be the frozen profile path: {_resolved(expected)}"
        )


def _profile(config: dict[str, Any]) -> contract.FrozenProfile:
    return contract.frozen_profile(str(config["profile"]))


def _require_execution_device(device: str, *, require_visibility: bool) -> None:
    if not require_visibility:
        return
    if str(device) != contract.EXECUTION_DEVICE:
        raise RuntimeError(
            f"device must be the frozen execution device: {contract.EXECUTION_DEVICE}"
        )
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if require_visibility and visible != contract.EXECUTION_CUDA_VISIBLE_DEVICES:
        raise RuntimeError(
            "CUDA_VISIBLE_DEVICES must equal the frozen execution value: "
            f"{contract.EXECUTION_CUDA_VISIBLE_DEVICES}"
        )


def _require_gate_output(config: dict[str, Any], actual: Path, filename: str) -> None:
    profile = _profile(config)
    if profile.name != contract.SFT372_PROFILE:
        return
    value = _resolved(actual)
    fixed = _resolved(profile.log_dir / "gates" / filename)
    if value == fixed:
        return
    try:
        relative = value.relative_to(_resolved(profile.log_dir))
    except ValueError as exc:
        raise RuntimeError("gate output is outside the frozen log root") from exc
    parts = relative.parts
    if (
        len(parts) != 3
        or _STAGING_DIRECTORY.fullmatch(parts[0]) is None
        or parts[1] != "reports"
        or parts[2] != filename
    ):
        raise RuntimeError("gate output is not an approved staging path")


def _require_pilot_dir(config: dict[str, Any], actual: Path) -> None:
    profile = _profile(config)
    if profile.name != contract.SFT372_PROFILE:
        return
    value = _resolved(actual)
    if value == _resolved(profile.pilot_dir):
        return
    try:
        relative = value.relative_to(_resolved(profile.log_dir))
    except ValueError as exc:
        raise RuntimeError("pilot directory is outside the frozen log root") from exc
    parts = relative.parts
    if (
        len(parts) != 2
        or _STAGING_DIRECTORY.fullmatch(parts[0]) is None
        or parts[1] != "pilot"
    ):
        raise RuntimeError("pilot directory is not an approved staging path")


def _fixed_gate_paths(
    config: dict[str, Any], args: argparse.Namespace
) -> dict[str, Path]:
    profile = _profile(config)
    result = {
        "structure": args.structure_report,
        "probability": args.probability_report,
        "memory": args.memory_report,
        "throughput": args.throughput_report,
        "pilot": args.pilot_report,
    }
    if profile.name == contract.SFT372_PROFILE:
        for name, path in result.items():
            _require_path(
                path, profile.log_dir / "gates" / f"{name}.json", f"{name} report"
            )
    return result


def _load(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(args.config)
    if _profile(config).name == contract.SFT372_PROFILE:
        _require_path(args.config, _profile(config).config_path, "config")
    signature = runtime_signature(config, config_path=args.config)
    return config, signature


def signature_command(args: argparse.Namespace) -> dict[str, Any]:
    config, signature = _load(args)
    _require_gate_output(config, args.output, "signature.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(signature, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return signature


def structure_command(args: argparse.Namespace) -> dict[str, Any]:
    config, signature = _load(args)
    _require_gate_output(config, args.output, "structure.json")
    return structure_gate(config, signature, output=args.output)


def probability_command(args: argparse.Namespace) -> dict[str, Any]:
    config, signature = _load(args)
    _require_execution_device(
        args.device, require_visibility=_profile(config).name == contract.SFT372_PROFILE
    )
    if _profile(config).name == contract.SFT372_PROFILE:
        require_gpu_identity(contract.EXPECTED_GPU_IDENTITY, args.device)
    _require_gate_output(config, args.output, "probability.json")
    return probability_gate(config, signature, output=args.output, device=args.device)


def memory_command(args: argparse.Namespace) -> dict[str, Any]:
    config, signature = _load(args)
    _require_execution_device(
        args.device, require_visibility=_profile(config).name == contract.SFT372_PROFILE
    )
    if _profile(config).name == contract.SFT372_PROFILE:
        require_gpu_identity(contract.EXPECTED_GPU_IDENTITY, args.device)
    _require_gate_output(config, args.output, "memory.json")
    return memory_gate(config, signature, output=args.output, device=args.device)


def throughput_command(args: argparse.Namespace) -> dict[str, Any]:
    config, signature = _load(args)
    _require_execution_device(
        args.device, require_visibility=_profile(config).name == contract.SFT372_PROFILE
    )
    if _profile(config).name == contract.SFT372_PROFILE:
        require_gpu_identity(contract.EXPECTED_GPU_IDENTITY, args.device)
    _require_gate_output(config, args.output, "throughput.json")
    return throughput_gate(config, signature, output=args.output, device=args.device)


def pilot_command(args: argparse.Namespace) -> dict[str, Any]:
    config, signature = _load(args)
    _require_execution_device(
        args.device, require_visibility=_profile(config).name == contract.SFT372_PROFILE
    )
    if _profile(config).name == contract.SFT372_PROFILE:
        require_gpu_identity(contract.EXPECTED_GPU_IDENTITY, args.device)
    _require_gate_output(config, args.output, "pilot.json")
    _require_pilot_dir(config, args.pilot_dir)
    return pilot_gate(
        config,
        signature,
        output=args.output,
        pilot_dir=args.pilot_dir,
        device=args.device,
    )


def cpu_gate_command(args: argparse.Namespace) -> dict[str, Any]:
    config, signature = _load(args)
    profile = _profile(config)
    _require_path(args.output, profile.log_dir / "cpu_gate.json", "CPU report")
    _require_path(args.log, profile.log_dir / "cpu_tests.log", "CPU test log")
    return run_cpu_gate(signature, output=args.output, log=args.log)


def cpu_check_command(args: argparse.Namespace) -> dict[str, Any]:
    config, signature = _load(args)
    profile = _profile(config)
    _require_path(args.report, profile.log_dir / "cpu_gate.json", "CPU report")
    _require_path(args.log, profile.log_dir / "cpu_tests.log", "CPU test log")
    return validate_cpu_gate_report(signature, args.report, expected_log=args.log)


def train_command(args: argparse.Namespace) -> dict[str, Any]:
    config, signature = _load(args)
    profile = _profile(config)
    if profile.name == contract.SFT372_PROFILE:
        _require_path(args.output_dir, profile.run_dir, "formal run directory")
    if profile.name == contract.SFT372_PROFILE:
        if args.cpu_report is None:
            raise RuntimeError("The SFT372 profile requires a CPU gate report.")
        _require_path(args.cpu_report, profile.log_dir / "cpu_gate.json", "CPU report")
        validate_cpu_gate_report(
            signature,
            args.cpu_report,
            expected_log=profile.log_dir / "cpu_tests.log",
        )
    _require_execution_device(
        args.device, require_visibility=profile.name == contract.SFT372_PROFILE
    )
    gate_paths = _fixed_gate_paths(config, args)
    current_gpu = (
        require_gpu_identity(contract.EXPECTED_GPU_IDENTITY, args.device)
        if profile.name == contract.SFT372_PROFILE
        else gpu_identity(args.device)
    )
    load_and_validate_gate_reports(
        config,
        signature,
        gate_paths,
        expected_gpu_identity=current_gpu,
    )
    return run_training(
        config,
        signature,
        output_dir=args.output_dir,
        device=args.device,
        resume=not args.no_resume,
        formal=profile.name == contract.SFT372_PROFILE,
    )


def verify_command(args: argparse.Namespace) -> dict[str, Any]:
    config, signature = _load(args)
    profile = _profile(config)
    if profile.name == contract.SFT372_PROFILE:
        _require_path(args.run_dir, profile.run_dir, "formal run directory")
        _require_path(
            args.output,
            profile.log_dir / "final_verification.json",
            "verification report",
        )
    if profile.name == contract.SFT372_PROFILE:
        if args.cpu_report is None:
            raise RuntimeError("The SFT372 profile requires a CPU gate report.")
        _require_path(args.cpu_report, profile.log_dir / "cpu_gate.json", "CPU report")
        validate_cpu_gate_report(
            signature,
            args.cpu_report,
            expected_log=profile.log_dir / "cpu_tests.log",
        )
    _require_execution_device(
        args.device, require_visibility=profile.name == contract.SFT372_PROFILE
    )
    gate_paths = _fixed_gate_paths(config, args)
    if profile.name == contract.SFT372_PROFILE:
        current_gpu = require_gpu_identity(contract.EXPECTED_GPU_IDENTITY, args.device)
        load_and_validate_gate_reports(
            config,
            signature,
            gate_paths,
            expected_gpu_identity=current_gpu,
        )
    else:
        load_and_validate_gate_reports(config, signature, gate_paths)
    return verify_run(
        config,
        signature,
        run_dir=args.run_dir,
        gate_paths=gate_paths,
        output=args.output,
        device=args.device,
        restore_recovery=profile.name == contract.SFT372_PROFILE,
    )


def _add_common(command: argparse.ArgumentParser) -> None:
    command.add_argument("--config", type=Path, required=True)


def _add_gpu_report(command: argparse.ArgumentParser) -> None:
    _add_common(command)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--device", default="cuda:0")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ksllm4rec-rloo-dapo")
    commands = parser.add_subparsers(dest="command", required=True)

    signature = commands.add_parser("signature")
    _add_common(signature)
    signature.add_argument("--output", type=Path, required=True)
    signature.set_defaults(handler=signature_command)

    structure = commands.add_parser("structure-gate")
    _add_common(structure)
    structure.add_argument("--output", type=Path, required=True)
    structure.set_defaults(handler=structure_command)

    probability = commands.add_parser("probability-gate")
    _add_gpu_report(probability)
    probability.set_defaults(handler=probability_command)

    memory = commands.add_parser("memory-gate")
    _add_gpu_report(memory)
    memory.set_defaults(handler=memory_command)

    throughput = commands.add_parser("throughput-gate")
    _add_gpu_report(throughput)
    throughput.set_defaults(handler=throughput_command)

    pilot = commands.add_parser("pilot-gate")
    _add_gpu_report(pilot)
    pilot.add_argument("--pilot-dir", type=Path, required=True)
    pilot.set_defaults(handler=pilot_command)

    cpu_gate = commands.add_parser("cpu-gate")
    _add_common(cpu_gate)
    cpu_gate.add_argument("--output", type=Path, required=True)
    cpu_gate.add_argument("--log", type=Path, required=True)
    cpu_gate.set_defaults(handler=cpu_gate_command)

    cpu_check = commands.add_parser("cpu-check")
    _add_common(cpu_check)
    cpu_check.add_argument("--report", type=Path, required=True)
    cpu_check.add_argument("--log", type=Path, required=True)
    cpu_check.set_defaults(handler=cpu_check_command)

    train = commands.add_parser("train")
    _add_common(train)
    train.add_argument("--structure-report", type=Path, required=True)
    train.add_argument("--probability-report", type=Path, required=True)
    train.add_argument("--memory-report", type=Path, required=True)
    train.add_argument("--throughput-report", type=Path, required=True)
    train.add_argument("--pilot-report", type=Path, required=True)
    train.add_argument("--cpu-report", type=Path)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--device", default="cuda:0")
    train.add_argument("--no-resume", action="store_true")
    train.set_defaults(handler=train_command)

    verify = commands.add_parser("verify")
    _add_common(verify)
    verify.add_argument("--structure-report", type=Path, required=True)
    verify.add_argument("--probability-report", type=Path, required=True)
    verify.add_argument("--memory-report", type=Path, required=True)
    verify.add_argument("--throughput-report", type=Path, required=True)
    verify.add_argument("--pilot-report", type=Path, required=True)
    verify.add_argument("--cpu-report", type=Path)
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
    verify.add_argument("--device", default=contract.EXECUTION_DEVICE)
    verify.set_defaults(handler=verify_command)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.handler(args)
    print(
        json.dumps(
            result,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
