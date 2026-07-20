"""Strict configuration loading for named GRPO experiment profiles."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .profiles import FRONTIER_PROFILE


EXPECTED_VALUES: dict[str, Any] = {
    "schema_version": 3,
    "spec_version": "3.1",
    "model.base_model": "/home/lyc/models/OneReason-0.8B-pretrain-competition",
    "model.sft_adapter": "/home/lyc/REC_PROJECTS/KSLLM4REC/artifacts/sft/runs/full_epoch_001",
    "model.tokenizer": "/home/lyc/REC_PROJECTS/KSLLM4REC/artifacts/sft/runs/full_epoch_001",
    "model.policy_adapter": "default",
    "model.reference_adapter": "reference",
    "model.attention": "flash_attention_2",
    "model.dtype": "bfloat16",
    "model.disable_dropout": True,
    "data.source": "/home/lyc/Data/HF_kuaishou-llmrec-sft-baseline-0.91/train.jsonl",
    "data.task": "recommend",
    "data.groups": 6378,
    "data.positives": 18651,
    "data.cutoff_len": 16384,
    "trie.strategy": "baseline_all_system_prompt_response_sids",
    "trie.unique_sids": 768593,
    "trie.domain_a_nodes": 10198,
    "trie.domain_ab_nodes": 385077,
    "rollout.num_generations": 8,
    "rollout.max_completion_length": 32,
    "rollout.do_sample": True,
    "rollout.num_beams": 1,
    "rollout.temperature": 1.0,
    "rollout.top_k": 0,
    "rollout.top_p": 1.0,
    "rollout.retain_duplicates": True,
    "rollout.replay_buffer": False,
    "rollout.num_iterations": 1,
    "rollout.add_gt_probability": 0.5,
    "rollout.add_gt_replaces_index": 7,
    "reward.exact": 1.0,
    "reward.same_ab": 0.2,
    "reward.same_a": 0.05,
    "reward.same_domain": 0.01,
    "reward.normalization": "zscore",
    "reward.std_correction": 1,
    "reward.std_epsilon": 1e-4,
    "loss.clip_epsilon": 0.2,
    "loss.reference_beta": 0.02,
    "train.epochs": 2,
    "train.prompt_microbatch_size": 1,
    "train.gradient_accumulation_groups": 8,
    "train.learning_rate": 5e-6,
    "train.weight_decay": 0.0,
    "train.adam_beta1": 0.9,
    "train.adam_beta2": 0.999,
    "train.adam_epsilon": 1e-8,
    "train.max_grad_norm": 1.0,
    "train.scheduler": "cosine",
    "train.warmup_ratio": 0.03,
    "train.gradient_checkpointing": True,
    "train.bf16": True,
    "train.tf32": True,
    "train.seed": 42,
    "train.logging_steps": 10,
    "train.resume_save_steps": 100,
    "memory.chunk_candidates": [8, 4, 2, 1],
    "memory.max_reserved_gib": 20.0,
    "memory.min_free_gib": 20.5,
    "gates.signal_groups": 512,
    "gates.timing_groups": 32,
    "gates.max_projected_hours": 48.0,
    "evaluation.fixed_probe": "/home/lyc/REC_PROJECTS/KSLLM4REC/artifacts/orpo/data/all_tasks_32705/fixed_probe_1024.jsonl",
    "evaluation.num_beams": 16,
    "evaluation.do_sample": False,
    "evaluation.max_completion_length": 32,
    "output.groups_dir": "/home/lyc/REC_PROJECTS/KSLLM4REC/artifacts/grpo/data/recommend_groups_v3_1",
    "output.trie_dir": "/home/lyc/REC_PROJECTS/KSLLM4REC/artifacts/grpo/catalog/baseline_all_sids_v3_1",
    "output.run_dir": "/home/lyc/REC_PROJECTS/KSLLM4REC/artifacts/grpo/runs/sft_grpo_g8_forcedgt_p050_zscore_2epoch",
    "output.log_dir": "/home/lyc/REC_PROJECTS/KSLLM4REC/operation_logs/grpo/v3_1",
}

# The Frontier profile intentionally has the same algorithmic values as V3.1.
# Only frozen data/model identities, derived-artifact locations, and the timing
# gate differ.  Keeping this map explicit makes accidental parameter drift
# fail at config load time.
FRONTIER_EXPECTED_VALUES: dict[str, Any] = dict(EXPECTED_VALUES)
FRONTIER_EXPECTED_VALUES.update(
    {
        "schema_version": FRONTIER_PROFILE.schema_version,
        "spec_version": FRONTIER_PROFILE.spec_version,
        "profile": FRONTIER_PROFILE.name,
        "model.sft_adapter": str(FRONTIER_PROFILE.sft_adapter),
        "model.tokenizer": str(FRONTIER_PROFILE.tokenizer),
        "data.source": str(FRONTIER_PROFILE.source),
        "data.provenance": str(FRONTIER_PROFILE.provenance),
        "data.groups": FRONTIER_PROFILE.groups,
        "data.positives": FRONTIER_PROFILE.positives,
        "trie.strategy": FRONTIER_PROFILE.trie_strategy,
        "trie.unique_sids": FRONTIER_PROFILE.unique_sids,
        "trie.domain_a_nodes": FRONTIER_PROFILE.domain_a_nodes,
        "trie.domain_ab_nodes": FRONTIER_PROFILE.domain_ab_nodes,
        "gates.max_projected_hours": 72.0,
        "evaluation.fixed_probe": str(FRONTIER_PROFILE.fixed_probe),
        "output.groups_dir": str(FRONTIER_PROFILE.groups_dir),
        "output.trie_dir": str(FRONTIER_PROFILE.trie_dir),
        "output.run_dir": str(FRONTIER_PROFILE.run_dir),
        "output.log_dir": str(FRONTIER_PROFILE.log_dir),
    }
)


def _flatten(value: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, child in value.items():
        dotted = f"{prefix}.{key}" if prefix else key
        if isinstance(child, dict):
            result.update(_flatten(child, dotted))
        else:
            result[dotted] = child
    return result


def load_config(path: Path) -> dict[str, Any]:
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("GRPO config must be a mapping.")
    actual = _flatten(value)
    profile_name = actual.get("profile")
    if profile_name is None:
        expected_values = EXPECTED_VALUES
        profile_label = "Spec V3.1"
    elif profile_name == FRONTIER_PROFILE.name:
        expected_values = FRONTIER_EXPECTED_VALUES
        profile_label = f"profile {FRONTIER_PROFILE.name!r}"
    else:
        raise ValueError(
            f"Unknown GRPO profile {profile_name!r}; expected historical V3.1 "
            f"config or {FRONTIER_PROFILE.name!r}."
        )
    missing = sorted(set(expected_values) - set(actual))
    unknown = sorted(set(actual) - set(expected_values))
    mismatches = {
        key: {"expected": expected, "actual": actual.get(key)}
        for key, expected in expected_values.items()
        if key in actual and actual[key] != expected
    }
    if missing or unknown or mismatches:
        raise ValueError(
            f"GRPO config differs from {profile_label}: "
            f"missing={missing}, unknown={unknown}, mismatches={mismatches}"
        )
    return value
