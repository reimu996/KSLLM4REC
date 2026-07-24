"""Command-line entry points for gates, pilot, formal training, and verification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config
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


def _load(args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    config = load_config(args.config)
    signature = runtime_signature(config, config_path=args.config)
    return config, signature


def signature_command(args: argparse.Namespace) -> dict[str, Any]:
    _, signature = _load(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(signature, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return signature


def structure_command(args: argparse.Namespace) -> dict[str, Any]:
    config, signature = _load(args)
    return structure_gate(config, signature, output=args.output)


def probability_command(args: argparse.Namespace) -> dict[str, Any]:
    config, signature = _load(args)
    return probability_gate(
        config, signature, output=args.output, device=args.device
    )


def memory_command(args: argparse.Namespace) -> dict[str, Any]:
    config, signature = _load(args)
    return memory_gate(config, signature, output=args.output, device=args.device)


def throughput_command(args: argparse.Namespace) -> dict[str, Any]:
    config, signature = _load(args)
    return throughput_gate(
        config, signature, output=args.output, device=args.device
    )


def pilot_command(args: argparse.Namespace) -> dict[str, Any]:
    config, signature = _load(args)
    return pilot_gate(
        config,
        signature,
        output=args.output,
        pilot_dir=args.pilot_dir,
        device=args.device,
    )


def train_command(args: argparse.Namespace) -> dict[str, Any]:
    config, signature = _load(args)
    load_and_validate_gate_reports(
        config,
        signature,
        {
            "structure": args.structure_report,
            "probability": args.probability_report,
            "memory": args.memory_report,
            "throughput": args.throughput_report,
            "pilot": args.pilot_report,
        },
    )
    return run_training(
        config,
        signature,
        output_dir=args.output_dir,
        device=args.device,
        resume=not args.no_resume,
    )


def verify_command(args: argparse.Namespace) -> dict[str, Any]:
    config, signature = _load(args)
    return verify_run(
        config,
        signature,
        run_dir=args.run_dir,
        gate_paths={
            "structure": args.structure_report,
            "probability": args.probability_report,
            "memory": args.memory_report,
            "throughput": args.throughput_report,
            "pilot": args.pilot_report,
        },
        output=args.output,
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

    train = commands.add_parser("train")
    _add_common(train)
    train.add_argument("--structure-report", type=Path, required=True)
    train.add_argument("--probability-report", type=Path, required=True)
    train.add_argument("--memory-report", type=Path, required=True)
    train.add_argument("--throughput-report", type=Path, required=True)
    train.add_argument("--pilot-report", type=Path, required=True)
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
    verify.add_argument("--run-dir", type=Path, required=True)
    verify.add_argument("--output", type=Path, required=True)
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
