"""Controlled, auditable commands for base-to-ORPO training."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _print(value: object) -> None:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))


def _freeze_sources(args: argparse.Namespace) -> int:
    from .integrity import create_generation_lock

    value = create_generation_lock(
        args.lock,
        model_root=args.model,
        source_dataset=args.source,
        pid2sid_dir=args.pid2sid,
        explorer_caption_dir=args.caption,
        explorer_text_probe_dir=args.text_probe,
        explorer_recommend_probe_dir=args.recommend_probe,
        llamafactory_root=args.llamafactory,
    )
    _print(value)
    return 0


def _verify_sources(args: argparse.Namespace) -> int:
    from .integrity import verify_generation_lock

    _print(verify_generation_lock(args.lock))
    return 0


def _prepare_candidates(args: argparse.Namespace) -> int:
    from .integrity import verify_generation_lock
    from .pairs import prepare_pair_candidates

    integrity = verify_generation_lock(args.generation_lock)
    report = prepare_pair_candidates(
        args.source, args.pid2sid, args.caption, args.output_dir
    )
    report["generation_integrity"] = integrity
    _print(report)
    return 0


def _mine_pairs(args: argparse.Namespace) -> int:
    from .integrity import verify_generation_lock
    from .miner import mine_pair_candidates

    integrity = verify_generation_lock(args.generation_lock)
    report = mine_pair_candidates(
        args.candidates,
        args.model,
        args.output_dir,
        min_free_gib=args.min_free_gib,
    )
    report["generation_integrity"] = integrity
    _print(report)
    return 0


def _preflight(args: argparse.Namespace) -> int:
    from .preflight import run_preflight

    _print(
        run_preflight(
            args.model,
            args.data_dir,
            args.report,
            cutoff_len=args.cutoff_len,
        )
    )
    return 0


def _prepare_probe(args: argparse.Namespace) -> int:
    from .probe import prepare_fixed_probe

    _print(
        prepare_fixed_probe(
            args.baseline,
            args.text_to_sid,
            args.recommend,
            args.output,
            args.report,
        )
    )
    return 0


def _lock_training(args: argparse.Namespace) -> int:
    from .integrity import create_training_lock

    value = create_training_lock(
        args.lock,
        generation_lock=args.generation_lock,
        model_root=args.model,
        source_dataset=args.source,
        data_files=args.data_file,
        llamafactory_root=args.llamafactory,
    )
    _print(value)
    return 0


def _load_config(path: Path) -> tuple[dict, dict]:
    from omegaconf import OmegaConf

    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise TypeError("Resolved ORPO YAML must be a mapping.")
    custom = dict(config.pop("custom_orpo"))
    return config, custom


def _config_check(args: argparse.Namespace) -> int:
    import llamafactory

    from ksllm4rec_sft.integrity import verify_environment_lock
    from ksllm4rec_sft.manifest import atomic_write_json

    from .contract import validate_parsed_contract, validate_training_contract
    from .fingerprint import implementation_fingerprint
    from .integrity import verify_training_lock

    config, custom = _load_config(args.config)
    contract = validate_training_contract(config, custom, "config_check")
    integrity = verify_training_lock(
        args.artifact_lock,
        llamafactory_module_file=Path(llamafactory.__file__),
    )
    environment = verify_environment_lock(args.environment_lock)
    from llamafactory.hparams import get_train_args

    model_args, data_args, training_args, finetuning_args, _ = get_train_args(config)
    parsed = validate_parsed_contract(
        model_args,
        data_args,
        training_args,
        finetuning_args,
        stage="config_check",
    )
    report = {
        "status": "passed",
        "resolved_config": config,
        "custom_orpo": custom,
        "training_contract": contract,
        "parsed_contract": parsed,
        "input_integrity": integrity,
        "environment_integrity": environment,
        "implementation_fingerprint": implementation_fingerprint(_project_root()),
    }
    atomic_write_json(args.report, report)
    _print(report)
    return 0


def _checkpoint_snapshot(output_dir: Path) -> list[dict]:
    from ksllm4rec_sft.checkpoint import validate_checkpoint

    checkpoints = []
    for path in sorted(
        output_dir.glob("checkpoint-*"),
        key=lambda value: int(value.name.removeprefix("checkpoint-")),
    ):
        checkpoints.append(validate_checkpoint(path))
    return checkpoints


def _require_passed_gates(args: argparse.Namespace) -> dict:
    from .gates import verify_gpu_gates

    if args.log_root is None or args.gates_report is None:
        raise ValueError("The full run requires --log-root and --gates-report.")
    return verify_gpu_gates(
        args.log_root,
        args.gates_report,
        _project_root(),
        args.max_reserved_gib,
    )


def _train(args: argparse.Namespace) -> int:
    import llamafactory
    import torch

    from ksllm4rec_sft.checkpoint import resolve_resume_checkpoint
    from ksllm4rec_sft.integrity import verify_environment_lock
    from ksllm4rec_sft.manifest import RunManifest, run_command

    from .contract import (
        FULL_STAGE,
        GATE_CUTOFFS,
        GATE_DATASET_NAME,
        validate_parsed_contract,
        validate_training_contract,
    )
    from .fingerprint import implementation_fingerprint
    from .integrity import snapshot_files, verify_training_lock

    config, custom = _load_config(args.config)
    config["output_dir"] = str(args.output_dir.resolve())
    if args.stage in GATE_CUTOFFS:
        config["dataset"] = GATE_DATASET_NAME
        config["cutoff_len"] = GATE_CUTOFFS[args.stage]
        config["max_steps"] = 1
        config["save_total_limit"] = 1
        config["resume_from_checkpoint"] = None
        resume = None
        gates = None
    elif args.stage == FULL_STAGE:
        gates = _require_passed_gates(args)
        resume = resolve_resume_checkpoint(args.output_dir)
        config["resume_from_checkpoint"] = resume["path"] if resume else None
    else:
        raise ValueError(f"Unsupported controlled training stage: {args.stage}")
    contract = validate_training_contract(config, custom, args.stage)
    integrity = verify_training_lock(
        args.artifact_lock,
        llamafactory_module_file=Path(llamafactory.__file__),
    )
    environment = verify_environment_lock(args.environment_lock)
    fingerprint = implementation_fingerprint(_project_root())

    from llamafactory.hparams import get_train_args

    from .workflow import run_chunked_orpo

    model_args, data_args, training_args, finetuning_args, _ = get_train_args(config)
    parsed = validate_parsed_contract(
        model_args,
        data_args,
        training_args,
        finetuning_args,
        stage=args.stage,
    )
    manifest = RunManifest(
        args.manifest,
        {
            "status": "running",
            "stage": args.stage,
            "command": sys.argv,
            "config_path": str(args.config.resolve()),
            "resolved_config": config,
            "custom_orpo": custom,
            "training_contract": contract,
            "parsed_contract": parsed,
            "resume_checkpoint": resume,
            "gpu_gates": gates,
            "input_integrity": integrity,
            "environment_integrity": environment,
            "implementation_fingerprint": fingerprint,
            "python_executable": sys.executable,
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
            "git": run_command(["git", "rev-parse", "HEAD"], _project_root()),
            "llamafactory_git": run_command(
                ["git", "rev-parse", "HEAD"],
                Path("/home/lyc/REC_PROJECTS/LlamaFactory"),
            ),
            "nvidia_smi_before": run_command(["nvidia-smi"]),
        },
    )
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required for controlled ORPO training.")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_gib = free_bytes / 1024**3
        manifest.update(
            gpu_free_gib_before=free_gib,
            gpu_total_gib=total_bytes / 1024**3,
        )
        if free_gib < args.min_free_gib:
            raise RuntimeError(
                f"GPU safety gate failed: {free_gib:.3f} GiB free, "
                f"need {args.min_free_gib:.3f} GiB."
            )
        torch.cuda.reset_peak_memory_stats()
        result = run_chunked_orpo(
            model_args,
            data_args,
            training_args,
            finetuning_args,
            lm_chunk_size=int(custom["chunk_size"]),
        )
        peak_allocated = torch.cuda.max_memory_allocated() / 1024**3
        peak_reserved = torch.cuda.max_memory_reserved() / 1024**3
        if peak_reserved > args.max_reserved_gib:
            raise RuntimeError(
                f"GPU reserved-memory gate failed: {peak_reserved:.3f} GiB > "
                f"{args.max_reserved_gib:.3f} GiB."
            )
        output_files = snapshot_files(
            args.output_dir / name
            for name in (
                "adapter_config.json",
                "adapter_model.safetensors",
                "trainer_state.json",
                "train_results.json",
            )
        )
        checkpoints = _checkpoint_snapshot(args.output_dir)
        manifest.update(
            status="passed",
            result=result,
            output_artifacts=output_files,
            checkpoints=checkpoints,
            peak_memory_allocated_gib=peak_allocated,
            peak_memory_reserved_gib=peak_reserved,
            nvidia_smi_after=run_command(["nvidia-smi"]),
        )
        return 0
    except Exception as exc:
        manifest.update(
            status="failed",
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
            nvidia_smi_after=run_command(["nvidia-smi"]),
        )
        raise


def _check_gates(args: argparse.Namespace) -> int:
    from .gates import verify_gpu_gates

    report = verify_gpu_gates(
        args.log_root,
        args.report,
        args.project_root,
        args.max_reserved_gib,
    )
    _print(report)
    return 0


def _verify(args: argparse.Namespace) -> int:
    from .verify import verify_full_run

    _print(
        verify_full_run(
            args.model,
            args.output_dir,
            args.log_root,
            args.report,
            args.artifact_lock,
            args.environment_lock,
            min_free_gib=args.min_free_gib,
            max_reserved_gib=args.max_reserved_gib,
        )
    )
    return 0


def _probe(args: argparse.Namespace) -> int:
    from .probe import evaluate_probe

    _print(
        evaluate_probe(
            args.model,
            args.probe,
            args.output_dir,
            adapter_path=args.adapter,
            batch_size=args.batch_size,
            min_free_gib=args.min_free_gib,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    freeze = sub.add_parser("freeze-sources")
    freeze.add_argument("--lock", type=Path, required=True)
    freeze.add_argument("--model", type=Path, required=True)
    freeze.add_argument("--source", type=Path, required=True)
    freeze.add_argument("--pid2sid", type=Path, required=True)
    freeze.add_argument("--caption", type=Path, required=True)
    freeze.add_argument("--text-probe", type=Path, required=True)
    freeze.add_argument("--recommend-probe", type=Path, required=True)
    freeze.add_argument("--llamafactory", type=Path, required=True)
    freeze.set_defaults(func=_freeze_sources)

    source_check = sub.add_parser("verify-sources")
    source_check.add_argument("--lock", type=Path, required=True)
    source_check.set_defaults(func=_verify_sources)

    candidates = sub.add_parser("prepare-candidates")
    candidates.add_argument("--generation-lock", type=Path, required=True)
    candidates.add_argument("--source", type=Path, required=True)
    candidates.add_argument("--pid2sid", type=Path, required=True)
    candidates.add_argument("--caption", type=Path, required=True)
    candidates.add_argument("--output-dir", type=Path, required=True)
    candidates.set_defaults(func=_prepare_candidates)

    mine = sub.add_parser("mine-pairs")
    mine.add_argument("--generation-lock", type=Path, required=True)
    mine.add_argument("--candidates", type=Path, required=True)
    mine.add_argument("--model", type=Path, required=True)
    mine.add_argument("--output-dir", type=Path, required=True)
    mine.add_argument("--min-free-gib", type=float, default=20.5)
    mine.set_defaults(func=_mine_pairs)

    preflight = sub.add_parser("preflight")
    preflight.add_argument("--model", type=Path, required=True)
    preflight.add_argument("--data-dir", type=Path, required=True)
    preflight.add_argument("--report", type=Path, required=True)
    preflight.add_argument("--cutoff-len", type=int, default=16_384)
    preflight.set_defaults(func=_preflight)

    probe_prepare = sub.add_parser("prepare-probe")
    probe_prepare.add_argument("--baseline", type=Path, required=True)
    probe_prepare.add_argument("--text-to-sid", type=Path, required=True)
    probe_prepare.add_argument("--recommend", type=Path, required=True)
    probe_prepare.add_argument("--output", type=Path, required=True)
    probe_prepare.add_argument("--report", type=Path, required=True)
    probe_prepare.set_defaults(func=_prepare_probe)

    lock = sub.add_parser("lock-training")
    lock.add_argument("--lock", type=Path, required=True)
    lock.add_argument("--generation-lock", type=Path, required=True)
    lock.add_argument("--model", type=Path, required=True)
    lock.add_argument("--source", type=Path, required=True)
    lock.add_argument("--data-file", type=Path, action="append", required=True)
    lock.add_argument("--llamafactory", type=Path, required=True)
    lock.set_defaults(func=_lock_training)

    config_check = sub.add_parser("config-check")
    config_check.add_argument("--config", type=Path, required=True)
    config_check.add_argument("--artifact-lock", type=Path, required=True)
    config_check.add_argument("--environment-lock", type=Path, required=True)
    config_check.add_argument("--report", type=Path, required=True)
    config_check.set_defaults(func=_config_check)

    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--artifact-lock", type=Path, required=True)
    train.add_argument("--environment-lock", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--stage", required=True)
    train.add_argument("--log-root", type=Path)
    train.add_argument("--gates-report", type=Path)
    train.add_argument("--min-free-gib", type=float, default=20.5)
    train.add_argument("--max-reserved-gib", type=float, default=20.0)
    train.set_defaults(func=_train)

    gates = sub.add_parser("check-gates")
    gates.add_argument("--log-root", type=Path, required=True)
    gates.add_argument("--report", type=Path, required=True)
    gates.add_argument("--project-root", type=Path, required=True)
    gates.add_argument("--max-reserved-gib", type=float, default=20.0)
    gates.set_defaults(func=_check_gates)

    verify = sub.add_parser("verify")
    verify.add_argument("--model", type=Path, required=True)
    verify.add_argument("--output-dir", type=Path, required=True)
    verify.add_argument("--log-root", type=Path, required=True)
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--artifact-lock", type=Path, required=True)
    verify.add_argument("--environment-lock", type=Path, required=True)
    verify.add_argument("--min-free-gib", type=float, default=20.5)
    verify.add_argument("--max-reserved-gib", type=float, default=20.0)
    verify.set_defaults(func=_verify)

    probe = sub.add_parser("evaluate-probe")
    probe.add_argument("--model", type=Path, required=True)
    probe.add_argument("--adapter", type=Path)
    probe.add_argument("--probe", type=Path, required=True)
    probe.add_argument("--output-dir", type=Path, required=True)
    probe.add_argument("--batch-size", type=int, default=4)
    probe.add_argument("--min-free-gib", type=float, default=20.5)
    probe.set_defaults(func=_probe)
    return parser


def main() -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("WANDB_DISABLED", "true")
    args = build_parser().parse_args()
    try:
        return args.func(args)
    finally:
        torch_module = sys.modules.get("torch")
        distributed = getattr(torch_module, "distributed", None)
        if distributed is not None and distributed.is_initialized():
            distributed.destroy_process_group()


if __name__ == "__main__":
    raise SystemExit(main())
