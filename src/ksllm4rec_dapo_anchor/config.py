"""Config template builder for DAPO-Anchor profiles.

Aligns to the same 12-section structure as the rloo_dapo SFT372 yaml, with
the only added section being `anchor:` (weight / batch_size / warmup_windows).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from . import contract


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
        "sample_canonical_max_logp_difference": (
            contract.SAMPLE_CANONICAL_MAX_LOGP_DIFF
        ),
        "canonical_replay_max_logp_difference": (
            contract.CANONICAL_REPLAY_MAX_LOGP_DIFF
        ),
        "num_generations": contract.GROUP_SIZE,
        "max_completion_length": 32,
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
        "overflow": "drop_and_audit",
        "shuffle_complete_groups": True,
        "max_unique_prompts_per_window": contract.EXPECTED_RECOMMEND_GROUPS,
        "source_order": "sha256_cycle_permutation",
    },
    "loss": {
        "objective": "dapo_anchor_asymmetric_clip",
        "reference_free": True,
        "decision_tokens_only": True,
        "aggregation": "minibatch_decision_token_mean",
        "clip_ratio_low": contract.CLIP_RATIO_LOW,
        "clip_ratio_high": contract.CLIP_RATIO_HIGH,
        "minibatch_groups": contract.MINIBATCH_GROUPS,
    },
    "anchor": {
        "weight": contract.ANCHOR_WEIGHT,
        "batch_size": contract.ANCHOR_BATCH_SIZE,
        "warmup_windows": contract.ANCHOR_WARMUP_WINDOWS,
    },
    "train": {
        "seed": 42,
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
        "gradient_checkpointing": contract.GRADIENT_CHECKPOINTING,
        "bf16": True,
        "tf32": True,
        "logging_windows": 10,
        "resume_save_updates": 4,
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
        "max_completion_length": 32,
    },
    "output": {
        "groups_dir": str(contract.GROUPS_DIR),
        "trie_dir": str(contract.TRIE_DIR),
        "run_dir": str(
            Path(contract.GROUPS_DIR).parent.parent
            / "runs/dapo_anchor_sft372_2effective_epochs"
        ),
        "log_dir": str(
            Path(contract.GROUPS_DIR).parent.parent.parent
            / "operation_logs/rloo/dapo_anchor_sft372"
        ),
    },
}


def build_config(
    *,
    seed: int = 42,
    learning_rate: float = contract.LEARNING_RATE,
    output_run_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build a fully-resolved config dict with optional overrides."""
    config = _CONFIG_TEMPLATE.copy()
    config["train"]["seed"] = int(seed)
    config["train"]["learning_rate"] = float(learning_rate)
    if output_run_dir is not None:
        config["output"]["run_dir"] = str(Path(output_run_dir).resolve())
    return config


__all__ = ["build_config"]
