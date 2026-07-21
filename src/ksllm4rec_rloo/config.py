"""Strict configuration loader for the approved reference-free RLOO run."""

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
        "candidate_chunk_size": contract.CHUNK_SIZE,
        "max_completion_length": 32,
        "do_sample": True,
        "num_beams": 1,
        "temperature": 1.0,
        "top_k": 0,
        "top_p": 1.0,
        "retain_duplicates": True,
        "replay_buffer": False,
        "num_iterations": 1,
    },
    "reward": {
        **dict(contract.REWARD_VALUES),
        "advantage": "rloo",
        "divide_by_std": False,
    },
    "loss": {
        "objective": "rloo_with_conditional_gt_set_anchor",
        "reference_free": True,
        "decision_tokens_only": True,
    },
    "anchor": {
        "trigger": "equal_rewards_non_exact",
        "target": "all_unique_gt",
        "calibration_groups": contract.CALIBRATION_GROUPS,
        "target_gradient_ratio": contract.ANCHOR_TARGET_GRADIENT_RATIO,
        "max_weight": contract.ANCHOR_MAX_WEIGHT,
        "min_grad_norm": contract.MIN_GRAD_NORM,
    },
    "train": {
        "epochs": contract.EPOCHS,
        "prompt_microbatch_size": 1,
        "gradient_accumulation_groups": contract.ACCUMULATION_GROUPS,
        "windows_per_epoch": contract.WINDOWS_PER_EPOCH,
        "total_windows": contract.TOTAL_WINDOWS,
        "learning_rate": contract.LEARNING_RATE,
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1.0e-8,
        "max_grad_norm": 1.0,
        "scheduler": "cosine",
        "warmup_windows": contract.WARMUP_WINDOWS,
        "gradient_checkpointing": True,
        "bf16": True,
        "tf32": True,
        "seed": 42,
        "logging_windows": 10,
        "resume_save_updates": 100,
        "shuffle": False,
    },
    "memory": {
        "loss_chunk_size": contract.CHUNK_SIZE,
        "gt_chunk_size": contract.CHUNK_SIZE,
        "max_reserved_gib": 20.0,
    },
    "gates": {
        "signal_groups": 512,
        "timing_groups": 256,
        "max_projected_hours": 72.0,
        "min_rloo_group_rate": 0.25,
        "max_logp_difference": 1.0e-5,
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
        "calibration_ids": str(contract.CALIBRATION_IDS),
    },
}

FORBIDDEN_KEY_PARTS = (
    "reference_adapter",
    "reference_beta",
    "forced_gt",
    "forced_mask",
    "add_gt",
    "clip_epsilon",
    "zscore",
    "std_correction",
    "std_epsilon",
    "kl_beta",
)


def approved_config() -> dict[str, Any]:
    """Return a mutable copy of the exact approved input configuration."""

    return deepcopy(APPROVED_CONFIG)


def _flatten(value: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    for key, child in value.items():
        if not isinstance(key, str) or not key:
            raise ValueError("RLOO config keys must be non-empty strings.")
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(child, Mapping):
            flattened.update(_flatten(child, dotted))
        else:
            flattened[dotted] = child
    return flattened


EXPECTED_VALUES = _flatten(APPROVED_CONFIG)


def validate_config(value: Mapping[str, Any]) -> dict[str, Any]:
    """Reject every behavioral or identity change from the approved Spec V2.0."""

    if not isinstance(value, Mapping):
        raise ValueError("RLOO config must be a mapping.")
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
            "RLOO config differs from approved Spec V2.0: "
            f"forbidden={forbidden}, missing={missing}, unknown={unknown}, "
            f"mismatches={mismatches}"
        )
    return deepcopy(dict(value))


def load_config(path: Path) -> dict[str, Any]:
    """Load YAML and apply the exact frozen contract."""

    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("RLOO config must be a mapping.")
    return validate_config(value)
