"""Frozen config builder for DAPO-Anchor-Multitask V1.0."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

from . import contract


def _arm_suffix(arm: str) -> str:
    return "" if arm == "C" else f"_arm_{arm.lower()}"


def _log_dir_for_arm(arm: str) -> Path:
    return (
        contract.PROJECT_ROOT
        / f"operation_logs/rloo/dapo_anchor_multitask_sft372_v1{_arm_suffix(arm)}"
    )


def _run_dir_for_arm(arm: str) -> Path:
    return (
        contract.PROJECT_ROOT
        / f"artifacts/rloo/runs/dapo_anchor_multitask_sft372_v1{_arm_suffix(arm)}"
    )


def _calibration_for_arm(arm: str) -> Path:
    return _log_dir_for_arm(arm) / "anchor_calibration.json"


_LOG_DIR = _log_dir_for_arm("C")
_RUN_DIR = _run_dir_for_arm("C")
_CALIBRATION = _calibration_for_arm("C")


def _arm_contract(arm: str) -> dict[str, Any]:
    if arm not in contract.EXPERIMENT_ARMS:
        raise ValueError(f"Unknown experiment arm: {arm!r}.")
    return {
        "arm": arm,
        "recommendation_enabled": True,
        "text_to_sid_enabled": arm == "C",
        "reward_profile": "old" if arm == "A" else "new",
    }


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
        "recommendation_groups": contract.EXPECTED_RECOMMENDATION_GROUPS,
        "recommendation_positives": contract.EXPECTED_RECOMMENDATION_POSITIVE_EDGES,
        "text_to_sid_raw_rows": contract.EXPECTED_TEXT_RAW_ROWS,
        "text_to_sid_groups": contract.EXPECTED_TEXT_GROUPS,
        "text_to_sid_positives": contract.EXPECTED_TEXT_POSITIVE_EDGES,
        "source_blocks": contract.SOURCE_BLOCKS,
        "cutoff_len": 16_384,
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
        "policy_snapshot": "shared_until_optimizer_window_commit",
        "rng_namespaces": ["recommendation", "item_text_to_sid"],
    },
    "sampling": {
        "source_blocks": contract.SOURCE_BLOCKS,
        "minimum_effective_groups_per_update": contract.MIN_EFFECTIVE_GROUPS_PER_UPDATE,
        "filter": "reward_range_positive",
        "overflow": "retain_all",
        "replenishment": False,
        "source_repeats": False,
        "shuffle": "sha256_task_group_id",
    },
    "reward": dict(contract.NEW_REWARD_VALUES),
    "loss": {
        "objective": "dapo_multitask_asymmetric_clip",
        "reference_free": True,
        "decision_tokens_only": True,
        "aggregation": "minibatch_decision_token_mean",
        "clip_ratio_low": contract.CLIP_RATIO_LOW,
        "clip_ratio_high": contract.CLIP_RATIO_HIGH,
        "minibatch_groups": contract.MINIBATCH_GROUPS,
    },
    "anchor": {
        "target": "negative_log_gt_set_probability_mass",
        "routes": ["no_exact_varied", "no_exact_flat"],
        "independent_optimizer_step": False,
        "merge_into": "final_policy_optimizer_step",
        "carry_without_rl": True,
        "final_auxiliary_flush": True,
        "calibration_source_blocks": contract.ANCHOR_CALIBRATION_BLOCKS,
        "min_calibration_source_blocks": contract.ANCHOR_MIN_CALIBRATION_BLOCKS,
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
        "source_epochs": contract.SOURCE_EPOCHS,
        "learning_rate": contract.LEARNING_RATE,
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1.0e-8,
        "max_grad_norm": contract.MAX_GRAD_NORM,
        "scheduler": "constant_after_policy_step_warmup",
        "warmup_policy_steps": contract.WARMUP_POLICY_STEPS,
        "warmup_steps_per_level": contract.WARMUP_STEPS_PER_LEVEL,
        "scheduler_step_unit": "policy_optimizer_step",
        "gradient_checkpointing": contract.GRADIENT_CHECKPOINTING,
        "bf16": True,
        "tf32": True,
        "checkpoint_source_blocks": contract.CHECKPOINT_SOURCE_BLOCK_INTERVAL,
        "retain_recovery_checkpoints": 2,
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
        "num_candidates": contract.EVALUATION_CANDIDATES,
        "require_unique_candidates": True,
        "do_sample": False,
        "max_completion_length": contract.MAX_COMPLETION_LENGTH,
        "formal_authority": ["R0", "R2", "total"],
    },
    "experiments": {
        "arms": list(contract.EXPERIMENT_ARMS),
        "active": _arm_contract("C"),
        "local_text_sid_pass64_min_delta": 0.0,
        "local_recommend_sid_pass64_max_drop": 0.005,
        "formal_r2_max_drop": 0.005,
    },
    "output": {
        "recommendation_groups_dir": str(contract.RECOMMENDATION_GROUPS_DIR),
        "text_to_sid_groups_dir": str(contract.TEXT_TO_SID_GROUPS_DIR),
        "trie_dir": str(contract.TRIE_DIR),
        "run_dir": str(_RUN_DIR),
        "log_dir": str(_LOG_DIR),
    },
}


def build_config(
    *,
    seed: int = 42,
    arm: str = "C",
    output_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    config = deepcopy(_CONFIG_TEMPLATE)
    config["train"]["seed"] = int(seed)
    config["experiments"]["active"] = _arm_contract(arm)
    reward = contract.OLD_REWARD_VALUES if arm == "A" else contract.NEW_REWARD_VALUES
    config["reward"] = dict(reward)
    config["anchor"]["calibration_path"] = str(_calibration_for_arm(arm))
    config["output"]["run_dir"] = str(_run_dir_for_arm(arm))
    config["output"]["log_dir"] = str(_log_dir_for_arm(arm))
    if output_run_dir is not None:
        config["output"]["run_dir"] = str(Path(output_run_dir).resolve())
    validate_config(config, allow_output_override=output_run_dir is not None)
    return config


def validate_config(
    config: Mapping[str, Any], *, allow_output_override: bool = False
) -> None:
    if config.get("schema_version") != contract.SCHEMA_VERSION:
        raise ValueError("Expected schema_version 3.")
    if config.get("spec_version") != contract.SPEC_VERSION:
        raise ValueError("Expected DAPO-Anchor-Multitask V1.0.")
    if config.get("profile") != contract.PROFILE:
        raise ValueError("Unexpected Multitask V1 profile.")
    active = config.get("experiments", {}).get("active", {})
    arm = active.get("arm")
    expected_arm = _arm_contract(str(arm))
    if active != expected_arm:
        raise ValueError("experiments.active differs from the frozen arm matrix.")
    expected_reward = (
        contract.OLD_REWARD_VALUES if arm == "A" else contract.NEW_REWARD_VALUES
    )
    for key, value in expected_reward.items():
        if config.get("reward", {}).get(key) != value:
            raise ValueError(f"Frozen config mismatch: reward.{key}")
    expected = {
        ("data", "source_blocks"): contract.SOURCE_BLOCKS,
        ("data", "recommendation_groups"): contract.EXPECTED_RECOMMENDATION_GROUPS,
        ("data", "text_to_sid_groups"): contract.EXPECTED_TEXT_GROUPS,
        ("rollout", "num_generations"): contract.GROUP_SIZE,
        ("rollout", "temperature"): contract.TEMPERATURE,
        ("sampling", "overflow"): "retain_all",
        ("sampling", "replenishment"): False,
        ("sampling", "source_repeats"): False,
        ("loss", "clip_ratio_low"): contract.CLIP_RATIO_LOW,
        ("loss", "clip_ratio_high"): contract.CLIP_RATIO_HIGH,
        ("loss", "minibatch_groups"): contract.MINIBATCH_GROUPS,
        ("anchor", "independent_optimizer_step"): False,
        ("anchor", "merge_into"): "final_policy_optimizer_step",
        ("anchor", "target_gradient_ratio"): contract.ANCHOR_TARGET_GRADIENT_RATIO,
        ("train", "source_epochs"): 1,
        ("train", "learning_rate"): contract.LEARNING_RATE,
        ("train", "scheduler_step_unit"): "policy_optimizer_step",
        ("evaluation", "num_candidates"): contract.EVALUATION_CANDIDATES,
        ("evaluation", "require_unique_candidates"): True,
    }
    for (section, key), value in expected.items():
        if config.get(section, {}).get(key) != value:
            raise ValueError(f"Frozen config mismatch: {section}.{key}")
    if not allow_output_override and Path(config["output"]["run_dir"]).resolve() != (
        _run_dir_for_arm(str(arm)).resolve()
    ):
        raise ValueError("Formal Multitask V1 run_dir was overridden.")
    if Path(config["output"]["log_dir"]).resolve() != _log_dir_for_arm(
        str(arm)
    ).resolve():
        raise ValueError("Formal Multitask V1 log_dir was overridden.")
    old_run = (
        contract.PROJECT_ROOT
        / "artifacts/rloo/runs/dapo_anchor_sft372_v2_2_2effective_epochs"
    ).resolve()
    if Path(config["output"]["run_dir"]).resolve() == old_run:
        raise ValueError("Multitask V1 may not write into the V2.2 run directory.")
    if Path(config["anchor"]["calibration_path"]).resolve() != _calibration_for_arm(
        str(arm)
    ).resolve():
        raise ValueError("Multitask V1 calibration_path was overridden.")


__all__ = ["build_config", "validate_config"]
