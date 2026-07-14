"""Executable contract for the approved OneReason LoRA SFT configuration."""

from __future__ import annotations

import math
from typing import Any


MODEL_PATH = "/home/lyc/models/OneReason-0.8B-pretrain-competition"
DATASET_DIR = "/home/lyc/REC_PROJECTS/KSLLM4REC/artifacts/sft/data/hf_baseline_091"
DATASET_NAME = "hf_kuaishou_llmrec_sft_baseline_0_91"
TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
GATE_CUTOFFS = {
    "gate_00512": 512,
    "gate_02048": 2048,
    "gate_08192": 8192,
    "gate_16384": 16384,
}
FULL_STAGE = "full_epoch_001"
CONFIG_CHECK_STAGE = "config_check"


def _normalise_targets(value: Any) -> set[str]:
    if isinstance(value, str):
        return {part.strip() for part in value.split(",") if part.strip()}
    return {str(part) for part in value}


def _same(actual: Any, expected: Any) -> bool:
    if isinstance(expected, float):
        try:
            return math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12)
        except (TypeError, ValueError):
            return False
    return actual == expected


def _requested_cutoff(run_stage: str) -> int:
    if run_stage in (FULL_STAGE, CONFIG_CHECK_STAGE):
        return 16384
    if run_stage in GATE_CUTOFFS:
        return GATE_CUTOFFS[run_stage]
    raise RuntimeError(f"Unapproved SFT run stage: {run_stage!r}")


def validate_training_contract(
    config: dict[str, Any], custom_loss: dict[str, Any], run_stage: str
) -> dict[str, Any]:
    """Reject any request that differs from the user-approved configuration."""

    cutoff = _requested_cutoff(run_stage)
    max_steps = 1 if run_stage in GATE_CUTOFFS else -1
    expected = {
        "model_name_or_path": MODEL_PATH,
        "trust_remote_code": True,
        "flash_attn": "fa2",
        "disable_gradient_checkpointing": False,
        "stage": "sft",
        "do_train": True,
        "finetuning_type": "lora",
        "lora_rank": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "additional_target": None,
        "pure_bf16": False,
        "dataset": DATASET_NAME,
        "dataset_dir": DATASET_DIR,
        "template": "qwen3_nothink",
        "cutoff_len": cutoff,
        "packing": True,
        "neat_packing": True,
        "train_on_prompt": False,
        "mask_history": False,
        "overwrite_cache": False,
        "preprocessing_num_workers": 8,
        "preprocessing_batch_size": 1000,
        "dataloader_num_workers": 2,
        "logging_strategy": "steps",
        "logging_steps": 5,
        "save_strategy": "steps",
        "save_steps": 256,
        "save_total_limit": 2,
        "save_only_model": False,
        "plot_loss": False,
        "overwrite_output_dir": False,
        "report_to": "none",
        "optim": "adamw_torch",
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1.0e-8,
        "max_grad_norm": 1.0,
        "learning_rate": 2.0e-4,
        "weight_decay": 0.001,
        "lr_scheduler_type": "cosine",
        "warmup_ratio": 0.03,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 1.0,
        "max_steps": max_steps,
        "bf16": True,
        "tf32": True,
        "gradient_checkpointing": True,
        "seed": 42,
        "data_seed": 42,
        "ddp_find_unused_parameters": False,
        "ddp_timeout": 180000000,
    }
    mismatches = [
        f"{key}: expected {expected_value!r}, got {config.get(key)!r}"
        for key, expected_value in expected.items()
        if not _same(config.get(key), expected_value)
    ]
    if _normalise_targets(config.get("lora_target", [])) != TARGET_MODULES:
        mismatches.append(
            "lora_target: expected "
            f"{sorted(TARGET_MODULES)!r}, got {sorted(_normalise_targets(config.get('lora_target', [])))!r}"
        )
    expected_custom = {"gamma": 2.0, "item_weight": 3.0, "chunk_size": 512}
    for key, expected_value in expected_custom.items():
        if not _same(custom_loss.get(key), expected_value):
            mismatches.append(
                f"custom_loss.{key}: expected {expected_value!r}, got {custom_loss.get(key)!r}"
            )
    resume = config.get("resume_from_checkpoint")
    if run_stage != FULL_STAGE and resume is not None:
        mismatches.append(
            f"resume_from_checkpoint: only {FULL_STAGE!r} may resume, got {resume!r}"
        )
    if mismatches:
        raise RuntimeError(
            "Training request violates the approved fixed configuration:\n- "
            + "\n- ".join(mismatches)
        )
    return {
        "status": "passed",
        "run_stage": run_stage,
        "requested_cutoff_len": cutoff,
        "max_steps": max_steps,
        "gradient_accumulation_steps": 8,
        "target_modules": sorted(TARGET_MODULES),
        "custom_loss": expected_custom,
    }


def validate_parsed_contract(
    model_args,
    data_args,
    training_args,
    finetuning_args,
    *,
    requested_cutoff: int,
) -> dict[str, Any]:
    """Verify important values after LLaMA-Factory has transformed arguments."""

    import torch

    checks = {
        "model_name_or_path": (
            str(model_args.model_name_or_path),
            MODEL_PATH,
        ),
        "flash_attn": (str(model_args.flash_attn), "fa2"),
        "compute_dtype": (model_args.compute_dtype, torch.bfloat16),
        "block_diag_attn": (model_args.block_diag_attn, True),
        "dataset": (list(data_args.dataset), [DATASET_NAME]),
        "template": (data_args.template, "qwen3_nothink"),
        "packing": (data_args.packing, True),
        "neat_packing": (data_args.neat_packing, True),
        "internal_cutoff_len": (data_args.cutoff_len, requested_cutoff - 1),
        "finetuning_type": (finetuning_args.finetuning_type, "lora"),
        "lora_rank": (finetuning_args.lora_rank, 32),
        "lora_alpha": (finetuning_args.lora_alpha, 32),
        "lora_dropout": (finetuning_args.lora_dropout, 0.05),
        "lora_target": (set(finetuning_args.lora_target), TARGET_MODULES),
        "bf16": (training_args.bf16, True),
        "tf32": (training_args.tf32, True),
        "gradient_checkpointing": (training_args.gradient_checkpointing, True),
        "per_device_train_batch_size": (
            training_args.per_device_train_batch_size,
            1,
        ),
        "gradient_accumulation_steps": (
            training_args.gradient_accumulation_steps,
            8,
        ),
        "world_size": (training_args.world_size, 1),
        "parallel_mode": (str(training_args.parallel_mode), "ParallelMode.DISTRIBUTED"),
    }
    mismatches = [
        f"{name}: expected {expected!r}, got {actual!r}"
        for name, (actual, expected) in checks.items()
        if not _same(actual, expected)
    ]
    if mismatches:
        raise RuntimeError(
            "Parsed LLaMA-Factory arguments violate the approved contract:\n- "
            + "\n- ".join(mismatches)
        )
    return {
        "status": "passed",
        "internal_cutoff_len": data_args.cutoff_len,
        "packed_sequence_len": data_args.cutoff_len + 1,
        "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
        "block_diag_attn": model_args.block_diag_attn,
        "parallel_mode": str(training_args.parallel_mode),
    }
