"""Frozen DAPO-Anchor V2.2 config builder and strict launcher validation."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from . import contract


_LOG_DIR = contract.PROJECT_ROOT / "operation_logs/rloo/dapo_anchor_sft372_v2_2"
_RUN_DIR = contract.PROJECT_ROOT / "artifacts/rloo/runs/dapo_anchor_sft372_v2_2_2effective_epochs"
_CALIBRATION = _LOG_DIR / "anchor_calibration.json"


_CONFIG_TEMPLATE: dict[str, Any] = {
    "schema_version": contract.SCHEMA_VERSION,
    "spec_version": contract.SPEC_VERSION,
    "profile": contract.PROFILE,
    "model": {
        "base_model": str(contract.BASE_MODEL),
        "sft_adapter": str(contract.SFT372_ADAPTER),
        "tokenizer": str(contract.SFT372_ADAPTER),
        "policy_adapter": contract.POLICY_ADAPTER_NAME,
        "attention": contract.ATTENTION,
        "dtype": contract.DTYPE,
        "disable_dropout": contract.DISABLE_DROPOUT,
        "lora_rank": contract.LORA_RANK,
        "lora_alpha": contract.LORA_ALPHA,
        "lora_dropout": contract.LORA_DROPOUT,
    },
    "data": {
        "source": str(contract.SOURCE_DATA),
        "provenance": str(contract.PROVENANCE),
        "task": "recommend",
        "groups": contract.EXPECTED_RECOMMEND_GROUPS,
        "positives": contract.EXPECTED_POSITIVE_EDGES,
        "cutoff_len": 16384,
        "prompt_mode": "direct_no_think",
    },
    "trie": {
        "strategy": "frontier_all_system_prompt_response_sids",
        "unique_sids": contract.EXPECTED_TRIE_LEAVES,
        "domain_a_nodes": contract.EXPECTED_DOMAIN_A_NODES,
        "domain_ab_nodes": contract.EXPECTED_DOMAIN_AB_NODES,
    },
    "rollout": {
        "temperature": contract.TEMPERATURE,
        "prompt_batch_size": contract.PROMPT_BATCH_SIZE,
        "max_active_sequences": contract.MAX_ACTIVE_SEQUENCES,
        "candidates_per_prompt_wave": contract.CANDIDATES_PER_PROMPT_WAVE,
        "sample_canonical_max_logp_difference": contract.SAMPLE_CANONICAL_MAX_LOGP_DIFF,
        "canonical_replay_max_logp_difference": contract.CANONICAL_REPLAY_MAX_LOGP_DIFF,
        "num_generations": contract.GROUP_SIZE,
        "max_completion_length": contract.MAX_COMPLETION_LENGTH,
        "do_sample": True,
        "num_beams": 1,
        "top_k": 0,
        "top_p": 1.0,
        "retain_duplicates": True,
        "use_kv_cache": True,
        "kv_cache_budget_gib": contract.KV_CACHE_BUDGET_GIB,
        "kv_cache_dtype": "bfloat16",
        "cache_fallback": "uncached_single_prompt",
        "replay_buffer": False,
        "num_iterations": contract.NUM_ITERATIONS,
    },
    "sampling": {
        "effective_groups_per_window": contract.EFFECTIVE_GROUPS_PER_WINDOW,
        "filter": "reward_range_positive",
        "anchor_candidate": "reward_flat_and_no_exact",
        "anchor_candidate_selection": "all",
        "overflow": "drop_and_audit",
        "shuffle_complete_groups": True,
        "max_unique_prompts_per_window": contract.EXPECTED_RECOMMEND_GROUPS,
        "source_order": "sha256_cycle_permutation",
    },
    "loss": {
        "objective": "dapo_anchor_v2_2_asymmetric_clip",
        "reference_free": True,
        "decision_tokens_only": True,
        "aggregation": "minibatch_decision_token_mean",
        "clip_ratio_low": contract.CLIP_RATIO_LOW,
        "clip_ratio_high": contract.CLIP_RATIO_HIGH,
        "minibatch_groups": contract.MINIBATCH_GROUPS,
    },
    "anchor": {
        "target": "negative_log_gt_set_probability_mass",
        "all_candidates": True,
        "optimizer_steps_per_window_max": 1,
        "warmup_windows": contract.ANCHOR_WARMUP_WINDOWS,
        "calibration_windows": contract.ANCHOR_CALIBRATION_WINDOWS,
        "min_calibration_windows": contract.ANCHOR_MIN_CALIBRATION_WINDOWS,
        "target_gradient_ratio": contract.ANCHOR_TARGET_GRADIENT_RATIO,
        "max_weight": contract.ANCHOR_MAX_WEIGHT,
        "gradient_epsilon": contract.ANCHOR_GRADIENT_EPSILON,
        "gt_completion_microbatch": contract.ANCHOR_GT_COMPLETION_MICROBATCH,
        "teacher_forcing_temperature": 1.0,
        "replay_max_logp_difference": contract.CANONICAL_REPLAY_MAX_LOGP_DIFF,
        "use_kv_cache": False,
        "calibration_path": str(_CALIBRATION),
    },
    "train": {
        "seed": 42,
        "effective_epochs": contract.EFFECTIVE_EPOCHS,
        "windows_per_effective_epoch": contract.WINDOWS_PER_EFFECTIVE_EPOCH,
        "total_windows": contract.TOTAL_WINDOWS,
        "total_rl_optimizer_updates": contract.TOTAL_RL_OPTIMIZER_UPDATES,
        "max_total_optimizer_updates": contract.MAX_TOTAL_OPTIMIZER_UPDATES,
        "rl_optimizer_updates_per_window": contract.OPTIMIZER_UPDATES_PER_WINDOW,
        "anchor_optimizer_updates_per_window_max": 1,
        "learning_rate": contract.LEARNING_RATE,
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1.0e-8,
        "max_grad_norm": contract.MAX_GRAD_NORM,
        "scheduler": "constant_after_linear_warmup",
        "warmup_windows": contract.WARMUP_WINDOWS,
        "scheduler_step_unit": "complete_rollout_window",
        "gradient_checkpointing": contract.GRADIENT_CHECKPOINTING,
        "bf16": True,
        "tf32": True,
        "logging_windows": 10,
        "resume_save_windows": 1,
        "num_iterations": contract.NUM_ITERATIONS,
    },
    "memory": {
        "loss_chunk_size": contract.LOSS_CHUNK_SIZE,
        "max_reserved_gib": contract.MAX_RESERVED_GIB,
        "kv_cache_budget_gib": contract.KV_CACHE_BUDGET_GIB,
        "model_num_layers": contract.MODEL_NUM_LAYERS,
        "model_num_kv_heads": contract.MODEL_NUM_KV_HEADS,
        "model_head_dim": contract.MODEL_HEAD_DIM,
        "cache_dtype_bytes": contract.MODEL_CACHE_DTYPE_BYTES,
        "max_observed_prompt_length": contract.MAX_OBSERVED_PROMPT_LENGTH,
    },
    "evaluation": {
        "fixed_probe": str(contract.FIXED_PROBE),
        "num_beams": 16,
        "do_sample": False,
        "max_completion_length": contract.MAX_COMPLETION_LENGTH,
    },
    "output": {
        "groups_dir": str(contract.GROUPS_DIR),
        "trie_dir": str(contract.TRIE_DIR),
        "run_dir": str(_RUN_DIR),
        "log_dir": str(_LOG_DIR),
    },
}


def build_config(
    *,
    seed: int = 42,
    learning_rate: float = contract.LEARNING_RATE,
    output_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = deepcopy(_CONFIG_TEMPLATE)
    config["train"]["seed"] = int(seed)
    config["train"]["learning_rate"] = float(learning_rate)
    if output_run_dir is not None:
        config["output"]["run_dir"] = str(Path(output_run_dir).resolve())
    validate_config(config, allow_output_override=output_run_dir is not None)
    return config


def validate_config(
    config: Mapping[str, Any], *, allow_output_override: bool = False
) -> None:
    """Reject semantically different runs before model/GPU initialization."""

    if config.get("schema_version") != 2 or config.get("spec_version") != contract.SPEC_VERSION:
        raise ValueError("Expected the DAPO-Anchor V2.2 config schema.")
    if config.get("profile") != contract.PROFILE:
        raise ValueError("Unexpected DAPO-Anchor V2.2 profile.")
    expected = {
        ("model", "sft_adapter"): str(contract.SFT372_ADAPTER),
        ("model", "lora_rank"): contract.LORA_RANK,
        ("model", "lora_alpha"): contract.LORA_ALPHA,
        ("rollout", "num_generations"): contract.GROUP_SIZE,
        ("rollout", "temperature"): contract.TEMPERATURE,
        ("sampling", "effective_groups_per_window"): contract.EFFECTIVE_GROUPS_PER_WINDOW,
        ("sampling", "anchor_candidate"): "reward_flat_and_no_exact",
        ("sampling", "anchor_candidate_selection"): "all",
        ("loss", "clip_ratio_low"): contract.CLIP_RATIO_LOW,
        ("loss", "clip_ratio_high"): contract.CLIP_RATIO_HIGH,
        ("loss", "minibatch_groups"): contract.MINIBATCH_GROUPS,
        ("anchor", "all_candidates"): True,
        ("anchor", "optimizer_steps_per_window_max"): 1,
        ("anchor", "warmup_windows"): contract.ANCHOR_WARMUP_WINDOWS,
        ("anchor", "calibration_windows"): contract.ANCHOR_CALIBRATION_WINDOWS,
        ("anchor", "min_calibration_windows"): contract.ANCHOR_MIN_CALIBRATION_WINDOWS,
        ("anchor", "target_gradient_ratio"): contract.ANCHOR_TARGET_GRADIENT_RATIO,
        ("anchor", "max_weight"): contract.ANCHOR_MAX_WEIGHT,
        ("anchor", "gt_completion_microbatch"): contract.ANCHOR_GT_COMPLETION_MICROBATCH,
        ("anchor", "teacher_forcing_temperature"): 1.0,
        ("anchor", "use_kv_cache"): False,
        ("train", "total_windows"): contract.TOTAL_WINDOWS,
        ("train", "total_rl_optimizer_updates"): contract.TOTAL_RL_OPTIMIZER_UPDATES,
        ("train", "max_total_optimizer_updates"): contract.MAX_TOTAL_OPTIMIZER_UPDATES,
        ("train", "rl_optimizer_updates_per_window"): contract.OPTIMIZER_UPDATES_PER_WINDOW,
        ("train", "anchor_optimizer_updates_per_window_max"): 1,
        ("train", "learning_rate"): contract.LEARNING_RATE,
        ("train", "warmup_windows"): contract.WARMUP_WINDOWS,
        ("memory", "max_reserved_gib"): contract.MAX_RESERVED_GIB,
    }
    for (section, key), value in expected.items():
        if config.get(section, {}).get(key) != value:
            raise ValueError(f"Frozen config mismatch: {section}.{key}")
    if not allow_output_override and Path(config["output"]["run_dir"]).resolve() != _RUN_DIR.resolve():
        raise ValueError("Formal V2.2 run_dir was overridden.")
    if Path(config["anchor"]["calibration_path"]).resolve() != _CALIBRATION.resolve():
        raise ValueError("V2.2 calibration_path was overridden.")


__all__ = ["build_config", "validate_config"]
