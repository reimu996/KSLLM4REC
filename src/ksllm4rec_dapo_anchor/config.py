"""Config template builder for DAPO-Anchor profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from . import contract


_CONFIG_TEMPLATE: dict[str, Any] = {
    "schema_version": contract.SCHEMA_VERSION,
    "spec_version": contract.SPEC_VERSION,
    "profile": contract.PROFILE,
    "data": {
        "source": str(contract.SOURCE_DATA),
        "provenance": str(contract.PROVENANCE),
        "cutoff_len": 32768,
        "source_rows": contract.EXPECTED_SOURCE_ROWS,
        "recommend_rows": contract.EXPECTED_RECOMMEND_ROWS,
        "groups": contract.EXPECTED_RECOMMEND_GROUPS,
        "positive_edges": contract.EXPECTED_POSITIVE_EDGES,
        "unique_positive_sids": contract.EXPECTED_UNIQUE_POSITIVE_SIDS,
    },
    "trie": {
        "unique_sids": contract.EXPECTED_TRIE_LEAVES,
        "domain_a_nodes": contract.EXPECTED_DOMAIN_A_NODES,
        "domain_ab_nodes": contract.EXPECTED_DOMAIN_AB_NODES,
    },
    "model": {
        "base": str(contract.BASE_MODEL),
        "base_sha256": contract.BASE_MODEL_SHA256,
        "source_sha256": contract.SOURCE_DATA_SHA256,
        "provenance_sha256": contract.PROVENANCE_SHA256,
        "groups_sha256": contract.GROUPS_SHA256,
        "trie_manifest_sha256": contract.TRIE_MANIFEST_SHA256,
        "fixed_probe_sha256": contract.FIXED_PROBE_SHA256,
        "lora_rank": contract.LORA_RANK,
        "lora_alpha": contract.LORA_ALPHA,
        "lora_dropout": contract.LORA_DROPOUT,
        "lora_target_modules": list(contract.LORA_TARGET_MODULES),
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
    },
    "loss": {
        "clip_ratio_low": contract.CLIP_RATIO_LOW,
        "clip_ratio_high": contract.CLIP_RATIO_HIGH,
    },
    "anchor": {
        "weight": contract.ANCHOR_WEIGHT,
        "batch_size": contract.ANCHOR_BATCH_SIZE,
        "warmup_windows": contract.ANCHOR_WARMUP_WINDOWS,
    },
    "sampling": {
        "effective_groups_per_window": contract.EFFECTIVE_GROUPS_PER_WINDOW,
        "max_unique_prompts_per_window": contract.EXPECTED_RECOMMEND_GROUPS,
    },
    "train": {
        "seed": 42,
        "effective_epochs": contract.EFFECTIVE_EPOCHS,
        "windows_per_effective_epoch": contract.WINDOWS_PER_EFFECTIVE_EPOCH,
        "total_windows": contract.TOTAL_WINDOWS,
        "learning_rate": contract.LEARNING_RATE,
        "warmup_windows": contract.WARMUP_WINDOWS,
        "max_grad_norm": contract.MAX_GRAD_NORM,
        "weight_decay": 0.0,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_epsilon": 1.0e-8,
        "scheduler": "constant_after_linear_warmup",
        "scheduler_step_unit": "rollout_window",
        "resume_save_updates": 4,
        "num_iterations": contract.NUM_ITERATIONS,
    },
    "memory": {
        "kv_cache_budget_gib": contract.KV_CACHE_BUDGET_GIB,
        "max_reserved_gib": contract.MAX_RESERVED_GIB,
    },
    "gpu": {
        "device": contract.EXECUTION_DEVICE,
        "cuda_visible_devices": contract.EXECUTION_CUDA_VISIBLE_DEVICES,
    },
    "output": {
        "groups_dir": str(contract.GROUPS_DIR),
        "trie_dir": str(contract.TRIE_DIR),
        "run_dir": str(
            Path(contract.GROUPS_DIR).parent.parent
            / "runs/dapo_anchor_pilot"
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
