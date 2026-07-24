"""Strict configuration loader for RLOO-DAPO-T1.2-KV Spec v2."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from . import contract


APPROVED_CONFIG: dict[str, Any] = {
    "schema_version": contract.SCHEMA_VERSION,
    "spec_version": contract.SPEC_VERSION,
    "profile": contract.PROFILE,
    "model": {
        "base_model": str(contract.BASE_MODEL),
        "sft_adapter": str(contract.SFT_ADAPTER),
        "tokenizer": str(contract.TOKENIZER),
        "policy_adapter": contract.POLICY_ADAPTER,
        "attention": "flash_attention_2",
        "dtype": "bfloat16",
        "disable_dropout": True,
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
        "num_generations": contract.GROUP_SIZE,
        "prompt_batch_size": contract.PROMPT_BATCH_SIZE,
        "max_active_sequences": contract.MAX_ACTIVE_SEQUENCES,
        "candidates_per_prompt_wave": contract.CANDIDATES_PER_PROMPT_WAVE,
        "max_completion_length": contract.MAX_COMPLETION_LENGTH,
        "do_sample": True,
        "num_beams": 1,
        "temperature": contract.TEMPERATURE,
        "top_k": 0,
        "top_p": 1.0,
        "retain_duplicates": True,
        "use_kv_cache": True,
        "kv_cache_budget_gib": contract.KV_CACHE_BUDGET_GIB,
        "kv_cache_dtype": "bfloat16",
        "cache_fallback": "uncached_single_prompt",
        "sample_canonical_max_logp_difference": (
            contract.SAMPLE_CANONICAL_MAX_LOGP_DIFF
        ),
        "canonical_replay_max_logp_difference": (
            contract.CANONICAL_REPLAY_MAX_LOGP_DIFF
        ),
        "replay_buffer": False,
        "num_iterations": contract.NUM_ITERATIONS,
    },
    "sampling": {
        "effective_groups_per_window": contract.EFFECTIVE_GROUPS_PER_WINDOW,
        "filter": "reward_range_positive",
        "overflow": "drop_and_audit",
        "shuffle_complete_groups": True,
        "max_unique_prompts_per_window": contract.EXPECTED_RECOMMEND_GROUPS,
        "source_order": "sha256_cycle_permutation",
    },
    "reward": {
        **dict(contract.REWARD_VALUES),
        "advantage": "rloo",
        "divide_by_std": False,
    },
    "loss": {
        "objective": "rloo_asymmetric_clip",
        "reference_free": True,
        "decision_tokens_only": True,
        "aggregation": "minibatch_decision_token_mean",
        "clip_ratio_low": contract.CLIP_RATIO_LOW,
        "clip_ratio_high": contract.CLIP_RATIO_HIGH,
        "minibatch_groups": contract.MINIBATCH_GROUPS,
    },
    "train": {
        "effective_epochs": contract.EFFECTIVE_EPOCHS,
        "windows_per_effective_epoch": contract.WINDOWS_PER_EFFECTIVE_EPOCH,
        "total_windows": contract.TOTAL_WINDOWS,
        "total_optimizer_updates": contract.TOTAL_OPTIMIZER_UPDATES,
        "optimizer_updates_per_window": contract.OPTIMIZER_UPDATES_PER_WINDOW,
        "learning_rate": contract.LEARNING_RATE,
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1.0e-8,
        "max_grad_norm": contract.MAX_GRAD_NORM,
        "scheduler": "constant_after_linear_warmup",
        "warmup_windows": contract.WARMUP_WINDOWS,
        "scheduler_step_unit": "rollout_window",
        "gradient_checkpointing": True,
        "bf16": True,
        "tf32": True,
        "seed": 42,
        "logging_windows": 10,
        "resume_save_updates": 100,
    },
    "memory": {
        "loss_chunk_size": contract.LOSS_CHUNK_SIZE,
        "max_reserved_gib": contract.MAX_RESERVED_GIB,
        "model_num_layers": contract.MODEL_NUM_LAYERS,
        "model_num_kv_heads": contract.MODEL_NUM_KV_HEADS,
        "model_head_dim": contract.MODEL_HEAD_DIM,
        "cache_dtype_bytes": contract.MODEL_CACHE_DTYPE_BYTES,
        "max_observed_prompt_length": contract.MAX_OBSERVED_PROMPT_LENGTH,
    },
    "gates": {
        "parity_groups": 64,
        "throughput_groups": 64,
        "end_to_end_windows": 1,
        "pilot_windows": 2,
        "min_rollout_speedup": 1.5,
        "min_window_speedup": 1.2,
        "require_all_legal": True,
        "require_all_finite": True,
    },
    "evaluation": {
        "fixed_probe": str(contract.FIXED_PROBE),
        "num_beams": 16,
        "do_sample": False,
        "max_completion_length": 32,
    },
    "output": {
        "groups_dir": str(contract.GROUPS_DIR),
        "trie_dir": str(contract.TRIE_DIR),
        "run_dir": str(contract.RUN_DIR),
        "log_dir": str(contract.LOG_DIR),
    },
}


FORBIDDEN_KEY_PARTS = (
    "anchor",
    "lambda0",
    "reference_adapter",
    "reference_beta",
    "forced_gt",
    "forced_mask",
    "add_gt",
    "gt_injection",
    "cosine",
    "zscore",
    "std_correction",
    "kl_beta",
)


def approved_config() -> dict[str, Any]:
    return deepcopy(APPROVED_CONFIG)


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("Config keys must be non-empty strings.")
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(child, Mapping):
            result.update(_flatten(child, dotted))
        else:
            result[dotted] = child
    return result


EXPECTED_VALUES = _flatten(APPROVED_CONFIG)


def validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("RLOO-DAPO config must be a mapping.")
    actual = _flatten(value)
    forbidden = sorted(
        key
        for key in actual
        if any(part in key.lower() for part in FORBIDDEN_KEY_PARTS)
        or key.lower().endswith(".kl")
    )
    missing = sorted(set(EXPECTED_VALUES) - set(actual))
    unknown = sorted(set(actual) - set(EXPECTED_VALUES))
    mismatches = {
        key: {"expected": expected, "actual": actual.get(key)}
        for key, expected in EXPECTED_VALUES.items()
        if key in actual and actual[key] != expected
    }
    if forbidden or missing or unknown or mismatches:
        raise ValueError(
            "RLOO-DAPO config differs from approved Spec v2: "
            f"forbidden={forbidden}, missing={missing}, unknown={unknown}, "
            f"mismatches={mismatches}"
        )
    return deepcopy(dict(value))


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("RLOO-DAPO config must be a mapping.")
    return validate_config(value)


__all__ = ["approved_config", "load_config", "validate_config"]
