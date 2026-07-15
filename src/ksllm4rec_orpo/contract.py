"""Frozen configuration contract for the approved base-to-ORPO run."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any


BASE_MODEL = Path("/home/lyc/models/OneReason-0.8B-pretrain-competition")
DATASET_DIR = Path(
    "/home/lyc/REC_PROJECTS/KSLLM4REC/artifacts/orpo/data/all_tasks_32705"
)
DATASET_NAME = "orpo_all_tasks_32705"
GATE_DATASET_NAME = "orpo_memory_gate_16384"
FULL_STAGE = "full_orpo_epoch_002"
GATE_CUTOFFS = {
    "gate_00512": 512,
    "gate_02048": 2_048,
    "gate_08192": 8_192,
    "gate_16384": 16_384,
}
EXPECTED_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def _require_equal(actual: Any, expected: Any, name: str) -> None:
    if actual != expected:
        raise ValueError(f"{name} must be {expected!r}, got {actual!r}.")


def _require_float(actual: Any, expected: float, name: str) -> None:
    if not math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError(f"{name} must be {expected}, got {actual!r}.")


def validate_training_contract(
    config: dict[str, Any], custom: dict[str, Any], stage: str
) -> dict[str, Any]:
    _require_equal(
        Path(config["model_name_or_path"]).resolve(), BASE_MODEL, "base model"
    )
    if config.get("adapter_name_or_path") not in (None, ""):
        raise ValueError("ORPO must start from the clean base without an adapter.")
    _require_equal(config.get("stage"), "dpo", "stage")
    _require_equal(config.get("do_train"), True, "do_train")
    _require_equal(config.get("finetuning_type"), "lora", "finetuning_type")
    _require_equal(config.get("pref_loss"), "orpo", "pref_loss")
    _require_float(config.get("pref_beta"), 0.1, "pref_beta")
    is_gate = stage in GATE_CUTOFFS
    expected_dataset = GATE_DATASET_NAME if is_gate else DATASET_NAME
    expected_cutoff = GATE_CUTOFFS.get(stage, 16_384)
    _require_equal(config.get("dataset"), expected_dataset, "dataset")
    _require_equal(
        Path(config["dataset_dir"]).resolve(), DATASET_DIR, "dataset_dir"
    )
    _require_equal(config.get("template"), "qwen3_nothink", "template")
    _require_equal(config.get("packing"), False, "packing")
    _require_equal(config.get("train_on_prompt"), False, "train_on_prompt")
    _require_equal(int(config.get("cutoff_len")), expected_cutoff, "cutoff_len")
    _require_equal(int(config.get("lora_rank")), 32, "lora_rank")
    _require_equal(int(config.get("lora_alpha")), 32, "lora_alpha")
    _require_float(config.get("lora_dropout"), 0.0, "lora_dropout")
    actual_targets = {
        item.strip() for item in str(config.get("lora_target", "")).split(",")
    }
    _require_equal(actual_targets, EXPECTED_TARGETS, "lora_target")
    _require_equal(config.get("optim"), "adamw_torch", "optimizer")
    _require_float(config.get("learning_rate"), 1.0e-4, "learning_rate")
    _require_float(config.get("weight_decay"), 0.001, "weight_decay")
    _require_float(config.get("warmup_ratio"), 0.03, "warmup_ratio")
    _require_equal(config.get("lr_scheduler_type"), "cosine", "lr_scheduler")
    _require_float(config.get("max_grad_norm"), 1.0, "max_grad_norm")
    _require_equal(int(config.get("per_device_train_batch_size")), 1, "batch_size")
    _require_equal(
        int(config.get("gradient_accumulation_steps")), 8, "gradient_accumulation"
    )
    _require_float(config.get("num_train_epochs"), 2.0, "num_train_epochs")
    _require_equal(config.get("bf16"), True, "bf16")
    _require_equal(config.get("tf32"), True, "tf32")
    _require_equal(config.get("flash_attn"), "fa2", "flash_attn")
    _require_equal(config.get("gradient_checkpointing"), True, "gradient_checkpointing")
    _require_equal(int(config.get("seed")), 42, "seed")
    _require_equal(int(config.get("data_seed")), 42, "data_seed")
    _require_equal(config.get("report_to"), "none", "report_to")
    _require_equal(int(custom.get("chunk_size")), 512, "custom chunk_size")
    _require_float(custom.get("max_reserved_gib"), 20.0, "max_reserved_gib")
    _require_float(custom.get("min_free_gib"), 20.5, "min_free_gib")

    if is_gate:
        _require_equal(int(config.get("max_steps")), 1, "gate max_steps")
    elif stage == FULL_STAGE:
        _require_equal(int(config.get("max_steps", -1)), -1, "full max_steps")
        _require_equal(config.get("save_strategy"), "epoch", "save_strategy")
    elif stage != "config_check":
        raise ValueError(f"Unknown controlled stage: {stage!r}")
    return {
        "status": "passed",
        "stage": stage,
        "clean_base_required": True,
        "reference_model_allowed": False,
        "requested_cutoff_len": int(config["cutoff_len"]),
        "epochs": float(config["num_train_epochs"]),
        "orpo_beta": float(config["pref_beta"]),
        "lm_chunk_size": int(custom["chunk_size"]),
    }


def validate_parsed_contract(
    model_args,
    data_args,
    training_args,
    finetuning_args,
    *,
    stage: str,
) -> dict[str, Any]:
    if model_args.adapter_name_or_path is not None:
        raise ValueError("Parsed model args unexpectedly load an adapter.")
    _require_equal(Path(model_args.model_name_or_path).resolve(), BASE_MODEL, "parsed model")
    is_gate = stage in GATE_CUTOFFS
    expected_dataset = GATE_DATASET_NAME if is_gate else DATASET_NAME
    expected_cutoff = GATE_CUTOFFS.get(stage, 16_384)
    _require_equal(data_args.dataset, [expected_dataset], "parsed dataset")
    _require_equal(Path(data_args.dataset_dir).resolve(), DATASET_DIR, "parsed dataset_dir")
    _require_equal(data_args.template, "qwen3_nothink", "parsed template")
    _require_equal(data_args.packing, False, "parsed packing")
    _require_equal(data_args.cutoff_len, expected_cutoff, "parsed cutoff")
    _require_equal(finetuning_args.stage, "dpo", "parsed stage")
    _require_equal(finetuning_args.pref_loss, "orpo", "parsed pref_loss")
    _require_equal(finetuning_args.use_ref_model, False, "parsed use_ref_model")
    _require_float(finetuning_args.pref_beta, 0.1, "parsed beta")
    _require_equal(finetuning_args.lora_rank, 32, "parsed lora_rank")
    _require_equal(finetuning_args.lora_alpha, 32, "parsed lora_alpha")
    _require_float(finetuning_args.lora_dropout, 0.0, "parsed lora_dropout")
    _require_equal(set(finetuning_args.lora_target), EXPECTED_TARGETS, "parsed targets")
    _require_equal(training_args.per_device_train_batch_size, 1, "parsed batch")
    _require_equal(training_args.gradient_accumulation_steps, 8, "parsed accumulation")
    _require_float(training_args.num_train_epochs, 2.0, "parsed epochs")
    _require_float(training_args.learning_rate, 1.0e-4, "parsed learning_rate")
    return {
        "status": "passed",
        "model": str(BASE_MODEL),
        "dataset": expected_dataset,
        "cutoff_len": data_args.cutoff_len,
        "reference_model_used": finetuning_args.use_ref_model,
        "epochs": training_args.num_train_epochs,
        "learning_rate": training_args.learning_rate,
    }
