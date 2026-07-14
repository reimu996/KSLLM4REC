"""Command line entrypoints for data preparation, preflight, and training."""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _prepare_data(args: argparse.Namespace) -> int:
    from .data import prepare_dataset

    report = prepare_dataset(args.source, args.output_dir)
    print(json.dumps(report.__dict__, ensure_ascii=False, indent=2))
    return 0


def _preflight(args: argparse.Namespace) -> int:
    from .preflight import tokenizer_preflight

    report = tokenizer_preflight(
        args.model,
        args.data,
        args.report,
        cutoff_len=args.cutoff_len,
        artifact_lock=args.artifact_lock,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _train(args: argparse.Namespace) -> int:
    import llamafactory
    import torch
    from omegaconf import OmegaConf

    from .checkpoint import resolve_resume_checkpoint
    from .contract import validate_parsed_contract, validate_training_contract
    from .fingerprint import implementation_fingerprint
    from .integrity import (
        snapshot_output_artifacts,
        verify_artifact_lock,
        verify_environment_lock,
    )
    from .manifest import RunManifest, run_command

    root = _project_root()
    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    custom = dict(config.pop("custom_loss"))
    config["output_dir"] = str(args.output_dir.resolve())
    if args.max_steps is not None:
        config["max_steps"] = args.max_steps
    if args.cutoff_len is not None:
        config["cutoff_len"] = args.cutoff_len
    config["overwrite_output_dir"] = args.overwrite_output_dir
    resume_checkpoint = None
    if args.stage == "full_epoch_001":
        resume_checkpoint = resolve_resume_checkpoint(args.output_dir)
        config["resume_from_checkpoint"] = (
            resume_checkpoint["path"] if resume_checkpoint is not None else None
        )
    else:
        config["resume_from_checkpoint"] = None
    contract = validate_training_contract(config, custom, args.stage)
    integrity = verify_artifact_lock(
        args.artifact_lock,
        llamafactory_module_file=Path(llamafactory.__file__),
    )
    environment = verify_environment_lock(args.environment_lock)
    fingerprint = implementation_fingerprint(root)

    from llamafactory.hparams import get_train_args

    from .workflow import run_focal_sft

    model_args, data_args, training_args, finetuning_args, generating_args = (
        get_train_args(config)
    )
    parsed_contract = validate_parsed_contract(
        model_args,
        data_args,
        training_args,
        finetuning_args,
        requested_cutoff=int(contract["requested_cutoff_len"]),
    )

    manifest = RunManifest(
        args.manifest,
        {
            "status": "running",
            "stage": args.stage,
            "command": sys.argv,
            "config_path": str(args.config.resolve()),
            "resolved_config": config,
            "custom_loss": custom,
            "training_contract": contract,
            "parsed_contract": parsed_contract,
            "resume_checkpoint": resume_checkpoint,
            "input_integrity": integrity,
            "environment_integrity": environment,
            "implementation_fingerprint": fingerprint,
            "python_executable": sys.executable,
            "conda_prefix": os.environ.get("CONDA_PREFIX"),
            "torch": {"version": torch.__version__, "cuda": torch.version.cuda},
            "git": run_command(["git", "rev-parse", "HEAD"], root),
            "llamafactory_git": run_command(
                ["git", "rev-parse", "HEAD"],
                Path("/home/lyc/REC_PROJECTS/LlamaFactory"),
            ),
            "nvidia_smi_before": run_command(["nvidia-smi"]),
        },
    )
    try:
        if torch.cuda.is_available():
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gib = free_bytes / 1024**3
            manifest.update(
                gpu_free_gib_before=free_gib, gpu_total_gib=total_bytes / 1024**3
            )
            if free_gib < args.min_free_gib:
                raise RuntimeError(
                    f"GPU safety gate failed: {free_gib:.3f} GiB free, {args.min_free_gib:.3f} GiB required."
                )

        result = run_focal_sft(
            model_args,
            data_args,
            training_args,
            finetuning_args,
            generating_args,
            focal_gamma=float(custom["gamma"]),
            item_weight=float(custom["item_weight"]),
            lm_chunk_size=int(custom["chunk_size"]),
        )
        peak_reserved = (
            torch.cuda.max_memory_reserved() / 1024**3
            if torch.cuda.is_available()
            else 0.0
        )
        if peak_reserved > args.max_reserved_gib:
            raise RuntimeError(
                f"GPU reserved-memory gate failed: {peak_reserved:.3f} GiB > {args.max_reserved_gib:.3f} GiB."
            )
        output_artifacts = snapshot_output_artifacts(args.output_dir)
        manifest.update(
            status="passed",
            result=result,
            output_artifacts=output_artifacts,
            peak_memory_allocated_gib=(
                torch.cuda.max_memory_allocated() / 1024**3
                if torch.cuda.is_available()
                else 0.0
            ),
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
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _config_check(args: argparse.Namespace) -> int:
    import llamafactory
    from omegaconf import OmegaConf

    from .contract import validate_parsed_contract, validate_training_contract
    from .fingerprint import implementation_fingerprint
    from .integrity import verify_artifact_lock, verify_environment_lock
    from .manifest import atomic_write_json

    config = OmegaConf.to_container(OmegaConf.load(args.config), resolve=True)
    custom = dict(config.pop("custom_loss"))
    contract = validate_training_contract(config, custom, "config_check")
    integrity = verify_artifact_lock(
        args.artifact_lock,
        llamafactory_module_file=Path(llamafactory.__file__),
    )
    environment = verify_environment_lock(args.environment_lock)

    from llamafactory.hparams import get_train_args

    model_args, data_args, training_args, finetuning_args, _ = get_train_args(config)
    parsed_contract = validate_parsed_contract(
        model_args,
        data_args,
        training_args,
        finetuning_args,
        requested_cutoff=int(contract["requested_cutoff_len"]),
    )
    report = {
        "status": "passed",
        "model_name_or_path": model_args.model_name_or_path,
        "flash_attn": model_args.flash_attn,
        "compute_dtype": str(model_args.compute_dtype),
        "block_diag_attn": model_args.block_diag_attn,
        "dataset": data_args.dataset,
        "template": data_args.template,
        "packing": data_args.packing,
        "neat_packing": data_args.neat_packing,
        "internal_cutoff_len": data_args.cutoff_len,
        "packed_sequence_len": data_args.cutoff_len + 1,
        "finetuning_type": finetuning_args.finetuning_type,
        "lora_rank": finetuning_args.lora_rank,
        "lora_alpha": finetuning_args.lora_alpha,
        "lora_dropout": finetuning_args.lora_dropout,
        "lora_target": sorted(finetuning_args.lora_target),
        "bf16": training_args.bf16,
        "world_size": training_args.world_size,
        "parallel_mode": str(training_args.parallel_mode),
        "custom_loss": custom,
        "training_contract": contract,
        "parsed_contract": parsed_contract,
        "input_integrity": integrity,
        "environment_integrity": environment,
        "implementation_fingerprint": implementation_fingerprint(_project_root()),
    }
    atomic_write_json(args.report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _verify(args: argparse.Namespace) -> int:
    import llamafactory
    import traceback

    from .integrity import verify_artifact_lock, verify_environment_lock
    from .manifest import atomic_write_json, now_iso
    from .verify import verify_adapter_files, verify_adapter_runtime

    report = {"status": "running", "created_at": now_iso()}
    atomic_write_json(args.report, report)
    try:
        report["input_integrity"] = verify_artifact_lock(
            args.artifact_lock,
            llamafactory_module_file=Path(llamafactory.__file__),
        )
        report["environment_integrity"] = verify_environment_lock(args.environment_lock)
        report["files"] = verify_adapter_files(
            args.output_dir,
            args.log_root,
            _project_root(),
            args.max_reserved_gib,
        )
        report["runtime"] = verify_adapter_runtime(
            args.model,
            args.output_dir,
            min_free_gib=args.min_free_gib,
        )
        report["status"] = "passed"
        report["updated_at"] = now_iso()
        atomic_write_json(args.report, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        report.update(
            status="failed",
            updated_at=now_iso(),
            error_type=type(exc).__name__,
            error=str(exc),
            traceback=traceback.format_exc(),
        )
        atomic_write_json(args.report, report)
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-data")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.set_defaults(func=_prepare_data)

    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--model", type=Path, required=True)
    preflight.add_argument("--data", type=Path, required=True)
    preflight.add_argument("--report", type=Path, required=True)
    preflight.add_argument("--cutoff-len", type=int, default=16384)
    preflight.add_argument("--artifact-lock", type=Path, required=True)
    preflight.set_defaults(func=_preflight)

    train = subparsers.add_parser("train")
    train.add_argument("--config", type=Path, required=True)
    train.add_argument("--artifact-lock", type=Path, required=True)
    train.add_argument("--environment-lock", type=Path, required=True)
    train.add_argument("--output-dir", type=Path, required=True)
    train.add_argument("--manifest", type=Path, required=True)
    train.add_argument("--stage", required=True)
    train.add_argument("--max-steps", type=int)
    train.add_argument("--cutoff-len", type=int)
    train.add_argument("--min-free-gib", type=float, default=20.5)
    train.add_argument("--max-reserved-gib", type=float, default=20.0)
    train.add_argument("--overwrite-output-dir", action="store_true")
    train.set_defaults(func=_train)

    gates = subparsers.add_parser("check-gates")
    gates.add_argument("--log-root", type=Path, required=True)
    gates.add_argument("--report", type=Path, required=True)
    gates.add_argument("--project-root", type=Path, required=True)
    gates.add_argument("--max-reserved-gib", type=float, default=20.0)
    gates.set_defaults(func=_check_gates)

    config_check = subparsers.add_parser("config-check")
    config_check.add_argument("--config", type=Path, required=True)
    config_check.add_argument("--artifact-lock", type=Path, required=True)
    config_check.add_argument("--environment-lock", type=Path, required=True)
    config_check.add_argument("--report", type=Path, required=True)
    config_check.set_defaults(func=_config_check)

    verify = subparsers.add_parser("verify")
    verify.add_argument("--model", type=Path, required=True)
    verify.add_argument("--output-dir", type=Path, required=True)
    verify.add_argument("--log-root", type=Path, required=True)
    verify.add_argument("--report", type=Path, required=True)
    verify.add_argument("--artifact-lock", type=Path, required=True)
    verify.add_argument("--environment-lock", type=Path, required=True)
    verify.add_argument("--min-free-gib", type=float, default=20.5)
    verify.add_argument("--max-reserved-gib", type=float, default=20.0)
    verify.set_defaults(func=_verify)
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
