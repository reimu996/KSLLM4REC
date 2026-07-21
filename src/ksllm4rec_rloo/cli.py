"""Reproducible command-line entry points for RLOO Spec V2.0."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .config import load_config
from .gates import (
    collect_input_sha256,
    expected_input_sha256,
    load_training_gates,
    run_calibration_gate,
    run_memory_gate,
    run_probability_gate,
    run_signal_gate,
    run_structure_gate,
    run_timing_gate,
)
from .trainer import write_calibration_ids


def _load_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _resolved_contract(path: Path) -> dict[str, Any]:
    report = _load_report(path)
    value = report.get("resolved_contract")
    if not isinstance(value, dict):
        raise ValueError(f"Calibration report has no resolved contract: {path}")
    return value


def config_check(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    actual = collect_input_sha256(config, args.groups, args.trie_dir)
    expected = expected_input_sha256()
    if actual != expected:
        raise RuntimeError(f"Frozen input SHA mismatch: expected={expected}, actual={actual}")
    return {
        "schema_version": config["schema_version"],
        "spec_version": config["spec_version"],
        "profile": config["profile"],
        "input_sha256": actual,
    }


def prepare_calibration(args: argparse.Namespace) -> dict[str, Any]:
    load_config(args.config)
    return write_calibration_ids(args.groups, args.output)


def structure_gate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    return run_structure_gate(
        config,
        groups_path=args.groups,
        trie_dir=args.trie_dir,
        calibration_ids_path=args.calibration_ids,
        output_path=args.output,
        config_path=args.config,
    )


def calibration_gate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    return run_calibration_gate(
        config,
        groups_path=args.groups,
        trie_dir=args.trie_dir,
        calibration_ids_path=args.calibration_ids,
        output_path=args.output,
        device=args.device,
        config_path=args.config,
    )


def probability_gate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    return run_probability_gate(
        config,
        groups_path=args.groups,
        trie_dir=args.trie_dir,
        calibration_ids_path=args.calibration_ids,
        output_path=args.output,
        resolved_contract=_resolved_contract(args.calibration_report),
        max_groups=args.max_groups,
        device=args.device,
        config_path=args.config,
    )


def memory_gate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    return run_memory_gate(
        config,
        groups_path=args.groups,
        trie_dir=args.trie_dir,
        calibration_ids_path=args.calibration_ids,
        output_path=args.output,
        resolved_contract=_resolved_contract(args.calibration_report),
        device=args.device,
        config_path=args.config,
    )


def signal_gate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    return run_signal_gate(
        config,
        groups_path=args.groups,
        trie_dir=args.trie_dir,
        calibration_ids_path=args.calibration_ids,
        output_dir=args.output_dir,
        output_path=args.output,
        resolved_contract=_resolved_contract(args.calibration_report),
        device=args.device,
        config_path=args.config,
    )


def timing_gate(args: argparse.Namespace) -> dict[str, Any]:
    config = load_config(args.config)
    return run_timing_gate(
        config,
        groups_path=args.groups,
        trie_dir=args.trie_dir,
        calibration_ids_path=args.calibration_ids,
        output_dir=args.output_dir,
        output_path=args.output,
        resolved_contract=_resolved_contract(args.calibration_report),
        device=args.device,
        config_path=args.config,
    )


def train(args: argparse.Namespace) -> dict[str, Any]:
    from .trainer import run_training

    config = load_config(args.config)
    gates = load_training_gates(
        config=config,
        groups_path=args.groups,
        trie_dir=args.trie_dir,
        calibration_ids_path=args.calibration_ids,
        structure_path=args.structure_report,
        calibration_path=args.calibration_report,
        probability_path=args.probability_report,
        memory_path=args.memory_report,
        signal_path=args.signal_report,
        timing_path=args.timing_report,
        config_path=args.config,
    )
    return run_training(
        config,
        groups_path=args.groups,
        trie_dir=args.trie_dir,
        output_dir=args.output_dir,
        resolved_contract=gates.resolved_contract,
        runtime_signature=gates.runtime_signature,
        resume=not args.no_resume,
        save_recovery=True,
        save_epochs=True,
        device=args.device,
    )


def probe(args: argparse.Namespace) -> dict[str, Any]:
    from .probe import run_fixed_probe

    return run_fixed_probe(
        load_config(args.config),
        policy_adapter_path=args.policy_adapter,
        trie_dir=args.trie_dir,
        groups_path=args.groups,
        calibration_ids_path=args.calibration_ids,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
    )


def verify(args: argparse.Namespace) -> dict[str, Any]:
    from .verify import verify_run

    return verify_run(
        load_config(args.config),
        config_path=args.config,
        groups_path=args.groups,
        trie_dir=args.trie_dir,
        calibration_ids_path=args.calibration_ids,
        run_dir=args.run_dir,
        probe_root=args.probe_root,
        gate_root=args.gate_root,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/rloo/frontier_sft_epoch2_lora64_g16.yaml"),
    )
    commands = parser.add_subparsers(dest="command", required=True)

    def add_inputs(command: argparse.ArgumentParser) -> None:
        command.add_argument("--groups", type=Path, required=True)
        command.add_argument("--trie-dir", type=Path, required=True)

    def add_bound_inputs(command: argparse.ArgumentParser) -> None:
        add_inputs(command)
        command.add_argument("--calibration-ids", type=Path, required=True)

    check = commands.add_parser("config-check")
    add_inputs(check)
    check.set_defaults(handler=config_check)

    prepare = commands.add_parser("prepare-calibration")
    prepare.add_argument("--groups", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.set_defaults(handler=prepare_calibration)

    structure = commands.add_parser("structure-gate")
    add_bound_inputs(structure)
    structure.add_argument("--output", type=Path, required=True)
    structure.set_defaults(handler=structure_gate)

    calibration = commands.add_parser("calibration-gate")
    add_bound_inputs(calibration)
    calibration.add_argument("--output", type=Path, required=True)
    calibration.add_argument("--device", default="cuda:0")
    calibration.set_defaults(handler=calibration_gate)

    def add_post_calibration(command: argparse.ArgumentParser) -> None:
        add_bound_inputs(command)
        command.add_argument("--calibration-report", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        command.add_argument("--device", default="cuda:0")

    probability = commands.add_parser("probability-gate")
    add_post_calibration(probability)
    probability.add_argument("--max-groups", type=int, default=1)
    probability.set_defaults(handler=probability_gate)

    memory = commands.add_parser("memory-gate")
    add_post_calibration(memory)
    memory.set_defaults(handler=memory_gate)

    signal = commands.add_parser("signal-gate")
    add_post_calibration(signal)
    signal.add_argument("--output-dir", type=Path, required=True)
    signal.set_defaults(handler=signal_gate)

    timing = commands.add_parser("timing-gate")
    add_post_calibration(timing)
    timing.add_argument("--output-dir", type=Path, required=True)
    timing.set_defaults(handler=timing_gate)

    formal = commands.add_parser("train")
    add_bound_inputs(formal)
    for name in (
        "structure",
        "calibration",
        "probability",
        "memory",
        "signal",
        "timing",
    ):
        formal.add_argument(f"--{name}-report", type=Path, required=True)
    formal.add_argument("--output-dir", type=Path, required=True)
    formal.add_argument("--device", default="cuda:0")
    formal.add_argument("--no-resume", action="store_true")
    formal.set_defaults(handler=train)

    fixed_probe = commands.add_parser("probe")
    add_bound_inputs(fixed_probe)
    fixed_probe.add_argument("--policy-adapter", type=Path, required=True)
    fixed_probe.add_argument("--output-dir", type=Path, required=True)
    fixed_probe.add_argument("--device", default="cuda:0")
    fixed_probe.set_defaults(handler=probe)

    final = commands.add_parser("verify")
    add_bound_inputs(final)
    final.add_argument("--run-dir", type=Path, required=True)
    final.add_argument("--probe-root", type=Path, required=True)
    final.add_argument("--gate-root", type=Path, required=True)
    final.set_defaults(handler=verify)
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
