"""Inference-only temperature A/B diagnostic for the final Frontier RLOO policy."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import torch
import yaml

from ksllm4rec_grpo.constraint import RecommendationGrammar
from ksllm4rec_grpo.data import RecommendationGroup, iter_groups
from ksllm4rec_grpo.prompt import encode_prompt
from ksllm4rec_grpo.trie import SidPrefixTrie
from ksllm4rec_orpo.data import Sid

from . import contract
from .config import load_config
from .fingerprint import runtime_signature
from .integrity import canonical_sha256, file_record, sha256_file
from .modeling import PolicyModel, load_policy_model
from .objective import rewards_and_advantages, sid_reward
from .rollout import RolloutCandidate, rollout_group, rollout_seed
from .scoring import score_completions
from .trainer import load_calibration_ids, set_global_seed


SCHEMA_VERSION = 1
ARM_BY_TEMPERATURE = {1.0: "t100", 1.2: "t120"}
TIER_ORDER = ("exact", "same_ab", "same_a", "same_domain", "other_domain")
EXPECTED_TOP_LEVEL = {
    "schema_version",
    "experiment_id",
    "purpose",
    "formal_rloo_config",
    "policy_adapter",
    "groups",
    "calibration_ids",
    "trie",
    "output_root",
    "sha256",
    "sampling",
    "reward",
    "safety",
    "decision",
}
EXPECTED_SHA_KEYS = {
    "formal_rloo_config",
    "adapter_model",
    "adapter_config",
    "groups",
    "calibration_ids",
    "calibration_group_ids",
    "trie_manifest",
}
EXPECTED_SAMPLING = {
    "temperatures": [1.0, 1.2],
    "groups": 512,
    "generations": 16,
    "chunk_size": 8,
    "max_completion_length": 32,
    "seed": 42,
    "epoch_index": 3,
    "top_k": 0,
    "top_p": 1.0,
    "retain_duplicates": True,
}
EXPECTED_SAFETY = {
    "inference_only": True,
    "max_logp_difference": 1.0e-5,
    "max_reserved_gib": 20.0,
    "require_all_legal": True,
    "require_all_finite": True,
    "forbid_gt_injection": True,
    "forbid_reward_conditioned_resampling": True,
}
EXPECTED_DECISION = {
    "min_informative_group_rate_delta": 0.05,
    "min_mean_unique_sid_delta": 0.5,
    "max_uniform_same_domain_rate_delta": -0.05,
    "min_any_exact_group_rate_delta": 0.0,
    "min_average_group_max_reward_delta": 0.0,
    "max_other_domain_slot_rate_delta": 0.02,
}


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], label: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{label} keys differ: missing={sorted(expected - actual)}, "
            f"unknown={sorted(actual - expected)}"
        )


def _mapping(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a mapping.")
    return value


def _path(value: Any, label: str, *, directory: bool = False) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} must be a non-empty path string.")
    path = Path(value).expanduser().resolve()
    exists = path.is_dir() if directory else path.is_file()
    if not exists:
        kind = "directory" if directory else "file"
        raise FileNotFoundError(f"{label} {kind} does not exist: {path}")
    return path


def _check_sha(path: Path, expected: Any, label: str) -> None:
    if not isinstance(expected, str) or len(expected) != 64:
        raise ValueError(f"{label} expected SHA256 must be 64 hexadecimal characters.")
    actual = sha256_file(path)
    if actual != expected:
        raise RuntimeError(f"{label} SHA256 mismatch: expected={expected}, actual={actual}")


def load_temperature_config(path: Path) -> dict[str, Any]:
    """Load and enforce the exact confirmed Temperature A/B contract."""

    config_path = Path(path).expanduser().resolve()
    value = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError("Temperature A/B config must be a mapping.")
    config = dict(value)
    _require_exact_keys(config, EXPECTED_TOP_LEVEL, "temperature config")
    if config["schema_version"] != SCHEMA_VERSION:
        raise ValueError("Unsupported temperature config schema_version.")
    if config["experiment_id"] != "frontier_epoch2_temperature_t100_t120_v1":
        raise ValueError("Unexpected temperature experiment_id.")
    if config["purpose"] != "inference_only_temperature_ab":
        raise ValueError("Temperature experiment must be inference-only.")

    hashes = _mapping(config["sha256"], "sha256")
    _require_exact_keys(hashes, EXPECTED_SHA_KEYS, "sha256")
    sampling = dict(_mapping(config["sampling"], "sampling"))
    if sampling != EXPECTED_SAMPLING:
        raise ValueError(
            f"sampling differs from confirmed Spec V1.1: {sampling!r}"
        )
    reward = dict(_mapping(config["reward"], "reward"))
    if reward != dict(contract.REWARD_VALUES):
        raise ValueError(f"reward differs from the frozen five tiers: {reward!r}")
    safety = dict(_mapping(config["safety"], "safety"))
    if safety != EXPECTED_SAFETY:
        raise ValueError(f"safety differs from confirmed Spec V1.1: {safety!r}")
    decision = dict(_mapping(config["decision"], "decision"))
    if decision != EXPECTED_DECISION:
        raise ValueError(f"decision differs from confirmed Spec V1.1: {decision!r}")

    formal_path = _path(config["formal_rloo_config"], "formal_rloo_config")
    adapter = _path(config["policy_adapter"], "policy_adapter", directory=True)
    groups = _path(config["groups"], "groups")
    calibration = _path(config["calibration_ids"], "calibration_ids")
    trie = _path(config["trie"], "trie", directory=True)
    _check_sha(formal_path, hashes["formal_rloo_config"], "formal_rloo_config")
    _check_sha(adapter / "adapter_model.safetensors", hashes["adapter_model"], "adapter_model")
    _check_sha(adapter / "adapter_config.json", hashes["adapter_config"], "adapter_config")
    _check_sha(groups, hashes["groups"], "groups")
    _check_sha(calibration, hashes["calibration_ids"], "calibration_ids")
    _check_sha(trie / "manifest.json", hashes["trie_manifest"], "trie_manifest")

    calibration_value = json.loads(calibration.read_text(encoding="utf-8"))
    if calibration_value.get("group_ids_sha256") != hashes["calibration_group_ids"]:
        raise RuntimeError("Calibration group ID SHA256 differs from the config.")
    formal = load_config(formal_path)
    if int(formal["rollout"]["num_generations"]) != sampling["generations"]:
        raise RuntimeError("Formal RLOO G differs from the diagnostic contract.")
    if int(formal["rollout"]["candidate_chunk_size"]) != sampling["chunk_size"]:
        raise RuntimeError("Formal RLOO chunk size differs from the diagnostic contract.")
    if int(formal["train"]["seed"]) != sampling["seed"]:
        raise RuntimeError("Formal RLOO seed differs from the diagnostic contract.")
    config["_config_path"] = str(config_path)
    return config


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _write_or_verify_json(path: Path, value: Any) -> None:
    if path.exists():
        if _load_json(path) != value:
            raise RuntimeError(f"Existing JSON output differs: {path}")
        return
    _atomic_json(path, value)


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected JSON object at {path}:{line_number}")
            rows.append(value)
    return rows


def _append_runtime(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {message}"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")
        handle.flush()
    print(line, file=sys.stderr, flush=True)


def _selected_inputs(
    config: Mapping[str, Any], max_groups: int
) -> tuple[dict[str, Any], list[RecommendationGroup], SidPrefixTrie]:
    if isinstance(max_groups, bool) or not 1 <= int(max_groups) <= 512:
        raise ValueError("max_groups must be within [1, 512].")
    formal = load_config(Path(str(config["formal_rloo_config"])))
    groups = list(iter_groups(Path(str(config["groups"]))))
    if len(groups) != int(formal["data"]["groups"]):
        raise RuntimeError(
            f"Expected {formal['data']['groups']} groups, found {len(groups)}."
        )
    selected_ids = load_calibration_ids(Path(str(config["calibration_ids"])), groups)
    by_id = {group.group_id: group for group in groups}
    selected = [by_id[group_id] for group_id in selected_ids[: int(max_groups)]]
    trie = SidPrefixTrie.load(
        Path(str(config["trie"])),
        expected_leaf_count=int(formal["trie"]["unique_sids"]),
    )
    return formal, selected, trie


def _git_head() -> str:
    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    )
    head = result.stdout.strip()
    if len(head) != 40 or any(character not in "0123456789abcdef" for character in head):
        raise RuntimeError(f"Invalid Git HEAD: {head!r}")
    return head


def diagnostic_runtime_signature(
    config: Mapping[str, Any],
    formal: dict[str, Any],
    selected: Sequence[RecommendationGroup],
    *,
    temperature: float,
) -> dict[str, Any]:
    """Bind every input that can change one diagnostic arm's output."""

    base = runtime_signature(
        formal,
        Path(str(config["groups"])),
        Path(str(config["trie"])),
        Path(str(config["calibration_ids"])),
        config_path=Path(str(config["formal_rloo_config"])),
    )
    adapter = Path(str(config["policy_adapter"]))
    common_inputs = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "temperature_config": file_record(Path(str(config["_config_path"]))),
        "formal_runtime": base,
        "policy_adapter": {
            "adapter_model": file_record(adapter / "adapter_model.safetensors"),
            "adapter_config": file_record(adapter / "adapter_config.json"),
        },
        "fixed_sampling": dict(config["sampling"]),
        "reward": dict(config["reward"]),
        "safety": dict(config["safety"]),
        "decision": dict(config["decision"]),
    }
    common = {"sha256": canonical_sha256(common_inputs), "inputs": common_inputs}
    arm_inputs = {
        "common_sha256": common["sha256"],
        "temperature": float(temperature),
        "sample_temperature": float(temperature),
        "replay_temperature": float(temperature),
        "selected_group_ids": [group.group_id for group in selected],
        "selected_group_ids_sha256": canonical_sha256(
            [group.group_id for group in selected]
        ),
    }
    return {
        "common": common,
        "arm": {"sha256": canonical_sha256(arm_inputs), "inputs": arm_inputs},
    }


def parameter_fingerprint(model: torch.nn.Module) -> dict[str, Any]:
    """Hash every trainable LoRA tensor without changing device placement."""

    digest = hashlib.sha256()
    tensors = 0
    numel = 0
    for name, parameter in sorted(model.named_parameters()):
        if not parameter.requires_grad:
            continue
        raw = parameter.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        metadata = json.dumps(
            [name, str(parameter.dtype), list(parameter.shape)],
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("ascii")
        digest.update(len(metadata).to_bytes(8, "big"))
        digest.update(metadata)
        digest.update(len(raw).to_bytes(8, "big"))
        digest.update(raw)
        tensors += 1
        numel += parameter.numel()
    if tensors == 0:
        raise RuntimeError("No trainable LoRA tensors were found for fingerprinting.")
    return {"sha256": digest.hexdigest(), "tensors": tensors, "numel": numel}


def _decision_stage(grammar: RecommendationGrammar, token_id: int) -> str:
    token = int(token_id)
    if token in grammar.domain_token_ids.values():
        return "domain"
    if grammar.a_offset <= token <= grammar.a_offset + 8_191:
        return "a"
    if grammar.b_offset <= token <= grammar.b_offset + 8_191:
        return "b"
    if grammar.c_offset <= token <= grammar.c_offset + 8_191:
        return "c"
    return "prefix"


def _finite_tree(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, dict):
        return all(_finite_tree(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_finite_tree(child) for child in value)
    return True


def _candidate_record(
    candidate: RolloutCandidate,
    replay_logps: Sequence[float],
    replay_decisions: Sequence[bool],
    reward: float,
    tier: str,
    grammar: RecommendationGrammar,
    group_id: str,
    epoch_index: int,
) -> dict[str, Any]:
    length = len(candidate.token_ids)
    if not (
        length
        == len(candidate.old_log_probs)
        == len(candidate.decision_mask)
        == len(candidate.legal_entropies)
        == len(candidate.legal_action_counts)
        == len(replay_logps)
        == len(replay_decisions)
    ):
        raise RuntimeError("Sampling and replay candidate lengths differ.")
    if tuple(bool(value) for value in replay_decisions) != candidate.decision_mask:
        raise RuntimeError("Sampling and replay decision masks differ.")
    differences = [
        abs(float(old) - float(new))
        for old, new in zip(candidate.old_log_probs, replay_logps, strict=True)
    ]
    decisions: list[dict[str, Any]] = []
    for position, is_decision in enumerate(candidate.decision_mask):
        count = int(candidate.legal_action_counts[position])
        entropy = float(candidate.legal_entropies[position])
        if is_decision:
            if count <= 1:
                raise RuntimeError("A decision token has fewer than two legal actions.")
            decisions.append(
                {
                    "token_position": position,
                    "stage": _decision_stage(grammar, candidate.token_ids[position]),
                    "legal_action_count": count,
                    "selected_token_id": int(candidate.token_ids[position]),
                    "sampled_logp": float(candidate.old_log_probs[position]),
                    "replay_logp": float(replay_logps[position]),
                    "entropy_nats": entropy,
                    "normalized_entropy": entropy / math.log(count),
                }
            )
        elif count != 1 or entropy != 0.0:
            raise RuntimeError("A deterministic token must have K=1 and entropy=0.")
    sid = grammar.parse(candidate.token_ids)
    if sid != candidate.sid:
        raise RuntimeError("Candidate SID differs from grammar parse result.")
    record = {
        "candidate_index": candidate.candidate_index,
        "seed": rollout_seed(group_id, epoch_index, candidate.candidate_index),
        "sid": candidate.sid.render(),
        "token_ids": list(candidate.token_ids),
        "reward": float(reward),
        "reward_tier": tier,
        "sampled_logps": [float(value) for value in candidate.old_log_probs],
        "replay_logps": [float(value) for value in replay_logps],
        "decision_mask": list(candidate.decision_mask),
        "legal_action_counts": list(candidate.legal_action_counts),
        "legal_entropies": [float(value) for value in candidate.legal_entropies],
        "decisions": decisions,
        "max_abs_sample_replay_logp_difference": max(differences, default=0.0),
    }
    if not _finite_tree(record):
        raise FloatingPointError("Candidate audit contains NaN or Inf.")
    return record


def group_derived_fields(candidate_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Recompute every group-level metric from the 16 candidate records."""

    candidates = list(candidate_records)
    if len(candidates) != 16:
        raise ValueError("A diagnostic group must contain exactly 16 candidates.")
    rewards = [float(candidate["reward"]) for candidate in candidates]
    tiers = [str(candidate["reward_tier"]) for candidate in candidates]
    sids = [str(candidate["sid"]) for candidate in candidates]
    entropy_values: list[float] = []
    normalized_values: list[float] = []
    maximum_difference = 0.0
    for candidate in candidates:
        sampled = [float(value) for value in candidate["sampled_logps"]]
        replayed = [float(value) for value in candidate["replay_logps"]]
        mask = [bool(value) for value in candidate["decision_mask"]]
        entropies = [float(value) for value in candidate["legal_entropies"]]
        action_counts = [int(value) for value in candidate["legal_action_counts"]]
        token_ids = [int(value) for value in candidate["token_ids"]]
        lengths = {
            len(sampled),
            len(replayed),
            len(mask),
            len(entropies),
            len(action_counts),
            len(token_ids),
        }
        if len(lengths) != 1:
            raise RuntimeError("Candidate token-level audit arrays have different lengths.")
        differences = [
            abs(left - right) for left, right in zip(sampled, replayed, strict=True)
        ]
        maximum_difference = max(maximum_difference, max(differences, default=0.0))
        for is_decision, entropy, count in zip(
            mask, entropies, action_counts, strict=True
        ):
            if is_decision:
                if count <= 1:
                    raise RuntimeError("A decision token has fewer than two legal actions.")
                entropy_values.append(entropy)
                normalized_values.append(entropy / math.log(count))
            elif count != 1 or entropy != 0.0:
                raise RuntimeError("A deterministic token must have K=1 and entropy=0.")
    unique = len(set(sids))
    derived = {
        "candidate_count": len(candidates),
        "reward_tiers": tiers,
        "reward_mean": math.fsum(rewards) / len(rewards),
        "reward_max": max(rewards),
        "informative_rloo": len(set(rewards)) > 1,
        "equal_reward": len(set(rewards)) == 1,
        "uniform_tier": tiers[0] if len(set(tiers)) == 1 else None,
        "any_exact": "exact" in tiers,
        "unique_sid_count": unique,
        "duplicate_slots": len(candidates) - unique,
        "all_same_sid": unique == 1,
        "decision_count": len(entropy_values),
        "entropy_sum_nats": math.fsum(entropy_values),
        "normalized_entropy_sum": math.fsum(normalized_values),
        "max_abs_sample_replay_logp_difference": maximum_difference,
        "all_legal": True,
        "all_finite": True,
        "gt_injection_count": 0,
        "backward_passes": 0,
        "optimizer_updates": 0,
    }
    if not _finite_tree(derived):
        raise FloatingPointError("Derived group audit contains NaN or Inf.")
    return derived


def _validate_candidate_audit(
    candidate: Mapping[str, Any],
    *,
    grammar: RecommendationGrammar,
    positives: Sequence[Sid],
    group_id: str,
    epoch_index: int,
    maximum_logp_difference: float,
) -> RolloutCandidate:
    expected_keys = {
        "candidate_index",
        "seed",
        "sid",
        "token_ids",
        "reward",
        "reward_tier",
        "sampled_logps",
        "replay_logps",
        "decision_mask",
        "legal_action_counts",
        "legal_entropies",
        "decisions",
        "max_abs_sample_replay_logp_difference",
    }
    _require_exact_keys(candidate, expected_keys, "candidate audit")
    index = int(candidate["candidate_index"])
    if candidate["seed"] != rollout_seed(group_id, epoch_index, index):
        raise RuntimeError("Candidate rollout seed differs from the frozen formula.")
    token_ids = tuple(int(value) for value in candidate["token_ids"])
    sampled = tuple(float(value) for value in candidate["sampled_logps"])
    replayed = tuple(float(value) for value in candidate["replay_logps"])
    decision_mask = tuple(bool(value) for value in candidate["decision_mask"])
    action_counts = tuple(int(value) for value in candidate["legal_action_counts"])
    entropies = tuple(float(value) for value in candidate["legal_entropies"])
    if len({len(token_ids), len(sampled), len(replayed), len(decision_mask), len(action_counts), len(entropies)}) != 1:
        raise RuntimeError("Candidate token-level audit arrays have different lengths.")
    sid = grammar.parse(token_ids)
    if sid.render() != candidate["sid"]:
        raise RuntimeError("Candidate SID differs from its legal token sequence.")
    expected_reward, expected_tier = sid_reward(sid, positives)
    if candidate["reward"] != expected_reward or candidate["reward_tier"] != expected_tier:
        raise RuntimeError("Candidate reward differs from the frozen reward function.")

    expected_decisions: list[dict[str, Any]] = []
    prefix: list[int] = []
    for position, token_id in enumerate(token_ids):
        allowed = [int(value) for value in grammar.allowed_next(prefix)]
        if token_id not in allowed:
            raise RuntimeError("Candidate token is absent from the legal prefix set.")
        count = len(allowed)
        is_decision = count > 1
        entropy = entropies[position]
        if action_counts[position] != count or decision_mask[position] != is_decision:
            raise RuntimeError("Candidate decision mask/action count differs from grammar.")
        if is_decision:
            maximum_entropy = math.log(count)
            if entropy < -1.0e-7 or entropy > maximum_entropy + 1.0e-6:
                raise RuntimeError("Candidate legal-action entropy is outside [0, log(K)].")
            expected_decisions.append(
                {
                    "token_position": position,
                    "stage": _decision_stage(grammar, token_id),
                    "legal_action_count": count,
                    "selected_token_id": token_id,
                    "sampled_logp": sampled[position],
                    "replay_logp": replayed[position],
                    "entropy_nats": entropy,
                    "normalized_entropy": entropy / maximum_entropy,
                }
            )
        elif entropy != 0.0:
            raise RuntimeError("A deterministic token has non-zero entropy.")
        prefix.append(token_id)
    if list(candidate["decisions"]) != expected_decisions:
        raise RuntimeError("Candidate decision details differ from token-level arrays.")
    difference = max(
        (abs(left - right) for left, right in zip(sampled, replayed, strict=True)),
        default=0.0,
    )
    if candidate["max_abs_sample_replay_logp_difference"] != difference:
        raise RuntimeError("Candidate stored max logp difference is not recomputable.")
    if difference > maximum_logp_difference:
        raise RuntimeError("Candidate sample/replay logp difference exceeds the contract.")
    if not _finite_tree(candidate):
        raise FloatingPointError("Candidate audit contains NaN or Inf.")
    return RolloutCandidate(
        candidate_index=index,
        sid=sid,
        token_ids=token_ids,
        old_log_probs=sampled,
        decision_mask=decision_mask,
        legal_entropies=entropies,
        legal_action_counts=action_counts,
    )


def _group_record(
    *,
    config: Mapping[str, Any],
    arm: str,
    temperature: float,
    group_index: int,
    group: RecommendationGroup,
    candidates: Sequence[RolloutCandidate],
    candidate_records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    positives = tuple(Sid.parse(value) for value in group.positive_sids)
    reward = rewards_and_advantages([item.sid for item in candidates], positives)
    if list(reward.tiers) != [item["reward_tier"] for item in candidate_records]:
        raise RuntimeError("Candidate reward tiers differ from the reward function.")
    derived = group_derived_fields(candidate_records)
    record = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "arm": arm,
        "temperature": float(temperature),
        "group_index": group_index,
        "group_id": group.group_id,
        "positive_sids": list(group.positive_sids),
        "candidates": list(candidate_records),
        **derived,
    }
    if not _finite_tree(record):
        raise FloatingPointError("Group audit contains NaN or Inf.")
    return record


def aggregate_audit(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate only deterministic fields from one arm's raw audit."""

    if not rows:
        raise ValueError("Cannot aggregate an empty audit.")
    groups = len(rows)
    slots = sum(int(row["candidate_count"]) for row in rows)
    tier_slots = Counter(
        str(candidate["reward_tier"])
        for row in rows
        for candidate in row["candidates"]
    )
    uniform = Counter(
        str(row["uniform_tier"])
        for row in rows
        if row["uniform_tier"] is not None
    )
    entropy_by_stage: dict[str, list[float]] = defaultdict(list)
    normalized_by_stage: dict[str, list[float]] = defaultdict(list)
    action_counts: list[int] = []
    for row in rows:
        for candidate in row["candidates"]:
            for decision in candidate["decisions"]:
                stage = str(decision["stage"])
                entropy_by_stage[stage].append(float(decision["entropy_nats"]))
                normalized_by_stage[stage].append(
                    float(decision["normalized_entropy"])
                )
                action_counts.append(int(decision["legal_action_count"]))
    entropy_count = len(action_counts)
    informative = sum(bool(row["informative_rloo"]) for row in rows)
    equal = sum(bool(row["equal_reward"]) for row in rows)
    exact_slots = tier_slots["exact"]
    any_exact = sum(bool(row["any_exact"]) for row in rows)
    all_same = sum(bool(row["all_same_sid"]) for row in rows)
    duplicate_slots = sum(int(row["duplicate_slots"]) for row in rows)
    metrics = {
        "groups": groups,
        "slots": slots,
        "informative_rloo_groups": informative,
        "informative_rloo_group_rate": informative / groups,
        "equal_reward_groups": equal,
        "equal_reward_group_rate": equal / groups,
        "uniform_tier_group_counts": {
            tier: uniform[tier] for tier in TIER_ORDER
        },
        "uniform_tier_group_rates": {
            tier: uniform[tier] / groups for tier in TIER_ORDER
        },
        "reward_tier_slot_counts": {
            tier: tier_slots[tier] for tier in TIER_ORDER
        },
        "reward_tier_slot_rates": {
            tier: tier_slots[tier] / slots for tier in TIER_ORDER
        },
        "average_reward": math.fsum(float(row["reward_mean"]) for row in rows)
        / groups,
        "average_group_max_reward": math.fsum(
            float(row["reward_max"]) for row in rows
        )
        / groups,
        "exact_slots": exact_slots,
        "exact_slot_rate": exact_slots / slots,
        "any_exact_groups": any_exact,
        "any_exact_group_rate": any_exact / groups,
        "mean_unique_sids_per_group": math.fsum(
            int(row["unique_sid_count"]) for row in rows
        )
        / groups,
        "duplicate_slots": duplicate_slots,
        "duplicate_slot_rate": duplicate_slots / slots,
        "all_same_sid_groups": all_same,
        "all_same_sid_group_rate": all_same / groups,
        "decision_count": entropy_count,
        "average_legal_action_entropy_nats": (
            math.fsum(
                float(row["entropy_sum_nats"])
                for row in rows
            )
            / entropy_count
            if entropy_count
            else 0.0
        ),
        "average_normalized_legal_action_entropy": (
            math.fsum(
                float(row["normalized_entropy_sum"])
                for row in rows
            )
            / entropy_count
            if entropy_count
            else 0.0
        ),
        "average_legal_action_count": (
            math.fsum(action_counts) / entropy_count if entropy_count else 0.0
        ),
        "entropy_by_stage": {
            stage: {
                "decisions": len(entropy_by_stage[stage]),
                "average_entropy_nats": math.fsum(entropy_by_stage[stage])
                / len(entropy_by_stage[stage]),
                "average_normalized_entropy": math.fsum(normalized_by_stage[stage])
                / len(normalized_by_stage[stage]),
            }
            for stage in sorted(entropy_by_stage)
        },
        "max_abs_sample_replay_logp_difference": max(
            float(row["max_abs_sample_replay_logp_difference"]) for row in rows
        ),
        "invalid_candidates": sum(
            0 if bool(row["all_legal"]) else int(row["candidate_count"])
            for row in rows
        ),
        "nonfinite_groups": sum(not bool(row["all_finite"]) for row in rows),
        "gt_injection_count": sum(int(row["gt_injection_count"]) for row in rows),
        "backward_passes": sum(int(row["backward_passes"]) for row in rows),
        "optimizer_updates": sum(int(row["optimizer_updates"]) for row in rows),
    }
    if sum(metrics["uniform_tier_group_counts"].values()) + informative != groups:
        raise RuntimeError("Uniform-tier and informative groups do not partition the arm.")
    if sum(metrics["reward_tier_slot_counts"].values()) != slots:
        raise RuntimeError("Reward tier slot counts do not sum to all slots.")
    return metrics


def _arm_paths(output_root: Path, arm: str) -> tuple[Path, Path]:
    return output_root / f"{arm}_audit.jsonl", output_root / f"{arm}_summary.json"


def run_arm(
    config_path: Path,
    *,
    temperature: float,
    output_root: Path | None = None,
    max_groups: int = 512,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Run one arm in a fresh process with no optimizer or backward pass."""

    config = load_temperature_config(config_path)
    temperature = float(temperature)
    if temperature not in ARM_BY_TEMPERATURE:
        raise ValueError("temperature must be exactly 1.0 or 1.2.")
    arm = ARM_BY_TEMPERATURE[temperature]
    root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else Path(str(config["output_root"])).expanduser().resolve()
    )
    root.mkdir(parents=True, exist_ok=True)
    audit_path, summary_path = _arm_paths(root, arm)
    temporary_audit = audit_path.with_name(f".{audit_path.name}.tmp")
    for path in (audit_path, summary_path, temporary_audit):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite diagnostic output: {path}")
    runtime_path = root / "runtime.log"
    _append_runtime(
        runtime_path,
        f"arm={arm} event=start temperature={temperature} groups={max_groups} device={device}",
    )

    formal, selected, trie = _selected_inputs(config, max_groups)
    runtime_binding = diagnostic_runtime_signature(
        config, formal, selected, temperature=temperature
    )
    execution_git_head = _git_head()
    torch_device = torch.device(device)
    if torch_device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Temperature A/B model execution requires a CUDA device.")
    device_index = (
        torch.cuda.current_device()
        if torch_device.index is None
        else int(torch_device.index)
    )
    torch.cuda.set_device(device_index)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device_index)
    set_global_seed(int(config["sampling"]["seed"]))
    adapter = Path(str(config["policy_adapter"]))
    source_before = {
        "adapter_model": file_record(adapter / "adapter_model.safetensors"),
        "adapter_config": file_record(adapter / "adapter_config.json"),
    }
    started = time.monotonic()
    bundle: PolicyModel = load_policy_model(
        formal, device=device, policy_adapter_path=adapter
    )
    bundle.model.train()
    if any(parameter.grad is not None for parameter in bundle.model.parameters()):
        raise RuntimeError("Fresh policy unexpectedly contains gradients.")
    parameters_before = parameter_fingerprint(bundle.model)
    grammar = RecommendationGrammar(bundle.tokenizer, trie)
    cutoff = int(formal["data"]["cutoff_len"])
    epoch_index = int(config["sampling"]["epoch_index"])
    chunk_size = int(config["sampling"]["chunk_size"])
    maximum_length = int(config["sampling"]["max_completion_length"])
    maximum_logp_difference = float(config["safety"]["max_logp_difference"])

    with temporary_audit.open("x", encoding="utf-8") as audit_handle:
        with torch.inference_mode():
            for group_index, group in enumerate(selected):
                prompt_ids = encode_prompt(
                    bundle.tokenizer,
                    group.system,
                    group.prompt,
                    cutoff_len=cutoff,
                )
                candidates = rollout_group(
                    bundle.model,
                    prompt_ids,
                    grammar,
                    group_id=group.group_id,
                    epoch_index=epoch_index,
                    chunk_size=chunk_size,
                    num_candidates=int(config["sampling"]["generations"]),
                    max_completion_length=maximum_length,
                    device=device,
                    temperature=temperature,
                    collect_legal_stats=True,
                )
                positives = tuple(Sid.parse(value) for value in group.positive_sids)
                reward = rewards_and_advantages(
                    [candidate.sid for candidate in candidates], positives
                )
                replay_rows: dict[int, tuple[list[float], list[bool]]] = {}
                for start in range(0, len(candidates), chunk_size):
                    chunk = candidates[start : start + chunk_size]
                    scores = score_completions(
                        bundle.model,
                        prompt_ids,
                        [candidate.token_ids for candidate in chunk],
                        grammar,
                        device=device,
                        temperature=temperature,
                    )
                    for local_index, candidate in enumerate(chunk):
                        length = len(candidate.token_ids)
                        if not bool(scores.valid_mask[local_index, :length].all()):
                            raise RuntimeError("Replay valid mask lost a completion token.")
                        if bool(scores.valid_mask[local_index, length:].any()):
                            raise RuntimeError("Replay valid mask includes padding.")
                        replay_rows[candidate.candidate_index] = (
                            scores.log_probs[local_index, :length]
                            .detach()
                            .float()
                            .cpu()
                            .tolist(),
                            scores.decision_mask[local_index, :length].cpu().tolist(),
                        )
                candidate_records = []
                for candidate, tier in zip(candidates, reward.tiers, strict=True):
                    replay_logps, replay_decisions = replay_rows[
                        candidate.candidate_index
                    ]
                    candidate_records.append(
                        _candidate_record(
                            candidate,
                            replay_logps,
                            replay_decisions,
                            float(contract.REWARD_VALUES[tier]),
                            tier,
                            grammar,
                            group.group_id,
                            epoch_index,
                        )
                    )
                row = _group_record(
                    config=config,
                    arm=arm,
                    temperature=temperature,
                    group_index=group_index,
                    group=group,
                    candidates=candidates,
                    candidate_records=candidate_records,
                )
                if row["max_abs_sample_replay_logp_difference"] > maximum_logp_difference:
                    raise RuntimeError(
                        "Sampling/replay log-probability difference exceeds "
                        f"{maximum_logp_difference}: "
                        f"{row['max_abs_sample_replay_logp_difference']}"
                    )
                audit_handle.write(
                    json.dumps(
                        row,
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                )
                audit_handle.flush()
                if (group_index + 1) % 8 == 0 or group_index + 1 == len(selected):
                    _append_runtime(
                        runtime_path,
                        f"arm={arm} event=progress groups={group_index + 1}/{len(selected)}",
                    )
        os.fsync(audit_handle.fileno())
    os.replace(temporary_audit, audit_path)

    torch.cuda.synchronize(device_index)
    seconds = time.monotonic() - started
    peak_allocated = torch.cuda.max_memory_allocated(device_index) / 1024**3
    peak_reserved = torch.cuda.max_memory_reserved(device_index) / 1024**3
    parameters_after = parameter_fingerprint(bundle.model)
    grad_tensors = sum(
        parameter.grad is not None for parameter in bundle.model.parameters()
    )
    source_after = {
        "adapter_model": file_record(adapter / "adapter_model.safetensors"),
        "adapter_config": file_record(adapter / "adapter_config.json"),
    }
    if source_before != source_after:
        raise RuntimeError("Source adapter files changed during inference-only diagnostic.")
    if parameters_before != parameters_after:
        raise RuntimeError("In-memory LoRA parameters changed without authorization.")
    if grad_tensors != 0:
        raise RuntimeError("Inference-only diagnostic created parameter gradients.")
    if peak_reserved > float(config["safety"]["max_reserved_gib"]):
        raise RuntimeError(
            f"Peak reserved memory {peak_reserved:.6f} GiB exceeds 20 GiB."
        )

    rows = _load_jsonl(audit_path)
    metrics = aggregate_audit(rows)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "arm": arm,
        "temperature": temperature,
        "sample_temperature": temperature,
        "replay_temperature": temperature,
        "config": file_record(Path(str(config["_config_path"]))),
        "formal_rloo_config": file_record(Path(str(config["formal_rloo_config"]))),
        "runtime_signature": runtime_binding,
        "execution_git_head": execution_git_head,
        "audit": file_record(audit_path),
        "selected_group_ids_sha256": canonical_sha256(
            [row["group_id"] for row in rows]
        ),
        "metrics": metrics,
        "safety": {
            "torch_inference_mode": True,
            "backward_passes": 0,
            "optimizer_updates": 0,
            "scheduler_steps": 0,
            "anchor_evaluations": 0,
            "gt_injection_count": 0,
            "reward_conditioned_resamples": 0,
            "parameter_grad_tensors": grad_tensors,
            "parameters_before": parameters_before,
            "parameters_after": parameters_after,
            "source_adapter_before": source_before,
            "source_adapter_after": source_after,
            "all_legal": metrics["invalid_candidates"] == 0,
            "all_finite": metrics["nonfinite_groups"] == 0,
        },
        "runtime": {
            "device": str(torch_device),
            "gpu_name": torch.cuda.get_device_name(device_index),
            "seconds": seconds,
            "peak_allocated_gib": peak_allocated,
            "peak_reserved_gib": peak_reserved,
        },
    }
    _atomic_json(summary_path, summary)
    _append_runtime(
        runtime_path,
        f"arm={arm} event=complete seconds={seconds:.3f} peak_reserved_gib={peak_reserved:.6f}",
    )
    return summary


def _paired_checks(
    t100_rows: Sequence[Mapping[str, Any]],
    t120_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if len(t100_rows) != len(t120_rows):
        raise RuntimeError("A/B arm group counts differ.")
    same_sid = 0
    same_tier = 0
    same_reward = 0
    seeds = 0
    transitions: Counter[str] = Counter()
    branch_transitions: Counter[str] = Counter()
    jaccards: list[float] = []
    for left, right in zip(t100_rows, t120_rows, strict=True):
        if left["group_index"] != right["group_index"] or left["group_id"] != right["group_id"]:
            raise RuntimeError("A/B group ID/order mismatch.")
        left_candidates = left["candidates"]
        right_candidates = right["candidates"]
        if len(left_candidates) != len(right_candidates):
            raise RuntimeError("A/B candidate counts differ within a group.")
        for a, b in zip(left_candidates, right_candidates, strict=True):
            if a["candidate_index"] != b["candidate_index"]:
                raise RuntimeError("A/B candidate index mismatch.")
            if a["seed"] != b["seed"]:
                raise RuntimeError("A/B candidate seed mismatch.")
            seeds += 1
            same_sid += a["sid"] == b["sid"]
            same_tier += a["reward_tier"] == b["reward_tier"]
            same_reward += a["reward"] == b["reward"]
            transitions[f"{a['reward_tier']}->{b['reward_tier']}"] += 1
        branch_transitions[
            f"{'rloo' if left['informative_rloo'] else 'equal'}->"
            f"{'rloo' if right['informative_rloo'] else 'equal'}"
        ] += 1
        left_sids = {item["sid"] for item in left_candidates}
        right_sids = {item["sid"] for item in right_candidates}
        jaccards.append(len(left_sids & right_sids) / len(left_sids | right_sids))
    return {
        "matched_groups": len(t100_rows),
        "matched_candidate_cells": seeds,
        "same_seed_cells": seeds,
        "same_sid_cells": same_sid,
        "same_sid_rate": same_sid / seeds,
        "same_reward_tier_cells": same_tier,
        "same_reward_tier_rate": same_tier / seeds,
        "same_reward_cells": same_reward,
        "same_reward_rate": same_reward / seeds,
        "reward_tier_transitions": dict(sorted(transitions.items())),
        "group_branch_transitions": dict(sorted(branch_transitions.items())),
        "mean_group_sid_set_jaccard": math.fsum(jaccards) / len(jaccards),
    }


def build_comparison(
    config: Mapping[str, Any],
    t100_rows: Sequence[Mapping[str, Any]],
    t120_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    left = aggregate_audit(t100_rows)
    right = aggregate_audit(t120_rows)
    deltas = {
        "informative_rloo_group_rate": right["informative_rloo_group_rate"]
        - left["informative_rloo_group_rate"],
        "mean_unique_sids_per_group": right["mean_unique_sids_per_group"]
        - left["mean_unique_sids_per_group"],
        "uniform_same_domain_group_rate": right["uniform_tier_group_rates"][
            "same_domain"
        ]
        - left["uniform_tier_group_rates"]["same_domain"],
        "any_exact_group_rate": right["any_exact_group_rate"]
        - left["any_exact_group_rate"],
        "average_group_max_reward": right["average_group_max_reward"]
        - left["average_group_max_reward"],
        "other_domain_slot_rate": right["reward_tier_slot_rates"]["other_domain"]
        - left["reward_tier_slot_rates"]["other_domain"],
        "average_reward": right["average_reward"] - left["average_reward"],
        "duplicate_slot_rate": right["duplicate_slot_rate"]
        - left["duplicate_slot_rate"],
        "average_legal_action_entropy_nats": right[
            "average_legal_action_entropy_nats"
        ]
        - left["average_legal_action_entropy_nats"],
    }
    thresholds = dict(config["decision"])
    gates = {
        "informative_rloo_group_rate": deltas["informative_rloo_group_rate"]
        >= thresholds["min_informative_group_rate_delta"],
        "mean_unique_sids_per_group": deltas["mean_unique_sids_per_group"]
        >= thresholds["min_mean_unique_sid_delta"],
        "uniform_same_domain_group_rate": deltas[
            "uniform_same_domain_group_rate"
        ]
        <= thresholds["max_uniform_same_domain_rate_delta"],
        "any_exact_group_rate": deltas["any_exact_group_rate"]
        >= thresholds["min_any_exact_group_rate_delta"],
        "average_group_max_reward": deltas["average_group_max_reward"]
        >= thresholds["min_average_group_max_reward_delta"],
        "other_domain_slot_rate": deltas["other_domain_slot_rate"]
        <= thresholds["max_other_domain_slot_rate_delta"],
        "sample_replay_logp": max(
            left["max_abs_sample_replay_logp_difference"],
            right["max_abs_sample_replay_logp_difference"],
        )
        <= float(config["safety"]["max_logp_difference"]),
        "all_legal": left["invalid_candidates"] == right["invalid_candidates"] == 0,
        "all_finite": left["nonfinite_groups"] == right["nonfinite_groups"] == 0,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "comparison": "temperature_1.2_minus_1.0",
        "fixed": {
            "groups": len(t100_rows),
            "generations": 16,
            "chunk_size": 8,
            "epoch_index": 3,
            "seed": 42,
        },
        "t100_metrics": left,
        "t120_metrics": right,
        "deltas": deltas,
        "paired": _paired_checks(t100_rows, t120_rows),
        "decision_thresholds": thresholds,
        "decision_gates": gates,
        "decision": "promising" if all(gates.values()) else "not_promising",
        "automatic_training_started": False,
    }


def compare_outputs(
    config_path: Path,
    *,
    output_root: Path | None = None,
    expected_groups: int = 512,
) -> dict[str, Any]:
    config = load_temperature_config(config_path)
    root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else Path(str(config["output_root"])).expanduser().resolve()
    )
    t100_rows = _load_jsonl(root / "t100_audit.jsonl")
    t120_rows = _load_jsonl(root / "t120_audit.jsonl")
    if len(t100_rows) != int(expected_groups) or len(t120_rows) != int(expected_groups):
        raise RuntimeError(
            f"Expected {expected_groups} groups per arm, got "
            f"{len(t100_rows)} and {len(t120_rows)}."
        )
    comparison = _comparison_with_runtime_binding(
        root, build_comparison(config, t100_rows, t120_rows)
    )
    path = root / "comparison.json"
    _write_or_verify_json(path, comparison)
    return comparison


def _comparison_with_runtime_binding(
    root: Path, comparison: dict[str, Any]
) -> dict[str, Any]:
    t100 = _load_json(root / "t100_summary.json")
    t120 = _load_json(root / "t120_summary.json")
    left_signature = t100.get("runtime_signature", {})
    right_signature = t120.get("runtime_signature", {})
    left_common = left_signature.get("common", {})
    right_common = right_signature.get("common", {})
    if left_common != right_common:
        raise RuntimeError("A/B arms do not share one common runtime signature.")
    left_head = t100.get("execution_git_head")
    right_head = t120.get("execution_git_head")
    if left_head != right_head:
        raise RuntimeError("A/B arms were not executed from the same Git HEAD.")
    return {
        **comparison,
        "runtime_binding": {
            "common_sha256": left_common.get("sha256"),
            "t100_arm_sha256": left_signature.get("arm", {}).get("sha256"),
            "t120_arm_sha256": right_signature.get("arm", {}).get("sha256"),
            "execution_git_head": left_head,
        },
    }


def _verify_arm(
    *,
    config: Mapping[str, Any],
    root: Path,
    arm: str,
    temperature: float,
    selected: Sequence[RecommendationGroup],
    grammar: RecommendationGrammar,
    expected_runtime_signature: Mapping[str, Any],
) -> dict[str, Any]:
    audit_path, summary_path = _arm_paths(root, arm)
    rows = _load_jsonl(audit_path)
    summary = _load_json(summary_path)
    if len(rows) != len(selected):
        raise RuntimeError(f"{arm} audit group count differs from expected selection.")
    epoch_index = int(config["sampling"]["epoch_index"])
    maximum_delta = float(config["safety"]["max_logp_difference"])
    for group_index, (row, group) in enumerate(zip(rows, selected, strict=True)):
        if row["group_index"] != group_index or row["group_id"] != group.group_id:
            raise RuntimeError(f"{arm} group selection/order mismatch at {group_index}.")
        if row["temperature"] != temperature or row["arm"] != arm:
            raise RuntimeError(f"{arm} temperature identity mismatch.")
        if row["positive_sids"] != list(group.positive_sids):
            raise RuntimeError(f"{arm} positive SID set mismatch at {group_index}.")
        positives = tuple(Sid.parse(value) for value in group.positive_sids)
        candidates = row["candidates"]
        if [item["candidate_index"] for item in candidates] != list(range(16)):
            raise RuntimeError(f"{arm} candidate indices differ from 0..15.")
        reconstructed: list[RolloutCandidate] = []
        for candidate in candidates:
            reconstructed.append(
                _validate_candidate_audit(
                    candidate,
                    grammar=grammar,
                    positives=positives,
                    group_id=group.group_id,
                    epoch_index=epoch_index,
                    maximum_logp_difference=maximum_delta,
                )
            )
        derived = group_derived_fields(candidates)
        for key, expected in derived.items():
            if row.get(key) != expected:
                raise RuntimeError(
                    f"{arm} derived group field {key!r} differs at {group_index}."
                )
        expected_group = _group_record(
            config=config,
            arm=arm,
            temperature=temperature,
            group_index=group_index,
            group=group,
            candidates=reconstructed,
            candidate_records=candidates,
        )
        if row != expected_group:
            raise RuntimeError(f"{arm} group audit schema/content differs at {group_index}.")
        if not _finite_tree(row):
            raise FloatingPointError(f"{arm} contains NaN/Inf at group {group_index}.")
    metrics = aggregate_audit(rows)
    if summary.get("metrics") != metrics:
        raise RuntimeError(f"{arm} summary metrics differ from raw audit aggregation.")
    if summary.get("sample_temperature") != temperature or summary.get(
        "replay_temperature"
    ) != temperature:
        raise RuntimeError(f"{arm} sample/replay temperature differs.")
    safety = summary.get("safety", {})
    zero_fields = (
        "backward_passes",
        "optimizer_updates",
        "scheduler_steps",
        "anchor_evaluations",
        "gt_injection_count",
        "reward_conditioned_resamples",
        "parameter_grad_tensors",
    )
    if any(safety.get(key) != 0 for key in zero_fields):
        raise RuntimeError(f"{arm} inference-only zero-counter invariant failed.")
    if safety.get("parameters_before") != safety.get("parameters_after"):
        raise RuntimeError(f"{arm} in-memory parameter fingerprint changed.")
    adapter = Path(str(config["policy_adapter"]))
    expected_source = {
        "adapter_model": file_record(adapter / "adapter_model.safetensors"),
        "adapter_config": file_record(adapter / "adapter_config.json"),
    }
    if (
        safety.get("source_adapter_before") != expected_source
        or safety.get("source_adapter_after") != expected_source
    ):
        raise RuntimeError(f"{arm} source adapter fingerprint differs from config.")
    if summary["runtime"]["peak_reserved_gib"] > float(
        config["safety"]["max_reserved_gib"]
    ):
        raise RuntimeError(f"{arm} peak reserved memory exceeds the contract.")
    if summary["audit"] != file_record(audit_path):
        raise RuntimeError(f"{arm} audit file record differs from current bytes.")
    if summary.get("runtime_signature") != expected_runtime_signature:
        raise RuntimeError(f"{arm} runtime signature differs from current inputs.")
    execution_head = summary.get("execution_git_head")
    if not isinstance(execution_head, str) or len(execution_head) != 40:
        raise RuntimeError(f"{arm} execution Git HEAD is invalid.")
    return {"rows": rows, "summary": summary, "metrics": metrics}


def verify_outputs(
    config_path: Path,
    *,
    output_root: Path | None = None,
    expected_groups: int = 512,
    smoke_root: Path | None = None,
    require_determinism: bool = False,
) -> dict[str, Any]:
    config = load_temperature_config(config_path)
    root = (
        Path(output_root).expanduser().resolve()
        if output_root is not None
        else Path(str(config["output_root"])).expanduser().resolve()
    )
    formal, selected, trie = _selected_inputs(config, expected_groups)
    left_runtime_signature = diagnostic_runtime_signature(
        config, formal, selected, temperature=1.0
    )
    right_runtime_signature = diagnostic_runtime_signature(
        config, formal, selected, temperature=1.2
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        Path(str(formal["model"]["tokenizer"])),
        trust_remote_code=True,
        local_files_only=True,
    )
    grammar = RecommendationGrammar(tokenizer, trie)
    left = _verify_arm(
        config=config,
        root=root,
        arm="t100",
        temperature=1.0,
        selected=selected,
        grammar=grammar,
        expected_runtime_signature=left_runtime_signature,
    )
    right = _verify_arm(
        config=config,
        root=root,
        arm="t120",
        temperature=1.2,
        selected=selected,
        grammar=grammar,
        expected_runtime_signature=right_runtime_signature,
    )
    expected_comparison = _comparison_with_runtime_binding(
        root, build_comparison(config, left["rows"], right["rows"])
    )
    comparison_path = root / "comparison.json"
    if _load_json(comparison_path) != expected_comparison:
        raise RuntimeError("comparison.json differs from independently recomputed output.")
    determinism: dict[str, Any] = {"required": False}
    if require_determinism:
        if smoke_root is None:
            raise ValueError("require_determinism needs --smoke-root.")
        smoke = Path(smoke_root).expanduser().resolve()
        smoke_report_path = smoke / "determinism.json"
        smoke_report = verify_determinism(
            left_root=smoke / "pass1_t100_then_t120",
            right_root=smoke / "pass2_t120_then_t100",
            expected_groups=8,
            output_path=smoke_report_path,
        )
        prefix_report_path = root / "first8_determinism.json"
        prefix_report = verify_determinism(
            left_root=smoke / "pass1_t100_then_t120",
            right_root=root,
            expected_groups=8,
            right_prefix=True,
            output_path=prefix_report_path,
        )
        determinism = {
            "required": True,
            "smoke_two_order_runs": smoke_report,
            "formal_first8_matches_smoke": prefix_report,
            "smoke_report": file_record(smoke_report_path),
            "formal_prefix_report": file_record(prefix_report_path),
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": config["experiment_id"],
        "passed": True,
        "groups_per_arm": expected_groups,
        "slots_per_arm": expected_groups * 16,
        "legal_candidates": expected_groups * 32,
        "all_finite": True,
        "sample_replay_temperatures": {"t100": 1.0, "t120": 1.2},
        "max_abs_sample_replay_logp_difference": max(
            left["metrics"]["max_abs_sample_replay_logp_difference"],
            right["metrics"]["max_abs_sample_replay_logp_difference"],
        ),
        "backward_passes": 0,
        "optimizer_updates": 0,
        "source_adapter_sha256": sha256_file(
            Path(str(config["policy_adapter"])) / "adapter_model.safetensors"
        ),
        "comparison_decision": expected_comparison["decision"],
        "determinism": determinism,
    }
    _write_or_verify_json(root / "verification.json", report)
    return report


def verify_determinism(
    *,
    left_root: Path,
    right_root: Path,
    expected_groups: int,
    right_prefix: bool = False,
    output_path: Path | None = None,
) -> dict[str, Any]:
    left_root = Path(left_root).expanduser().resolve()
    right_root = Path(right_root).expanduser().resolve()
    arms: dict[str, Any] = {}
    for arm in ("t100", "t120"):
        left_rows = _load_jsonl(left_root / f"{arm}_audit.jsonl")
        right_rows = _load_jsonl(right_root / f"{arm}_audit.jsonl")
        if right_prefix:
            right_rows = right_rows[:expected_groups]
        if len(left_rows) != expected_groups or len(right_rows) != expected_groups:
            raise RuntimeError(f"{arm} deterministic comparison group count differs.")
        left_payload = [dict(row) for row in left_rows]
        right_payload = [dict(row) for row in right_rows]
        if left_payload != right_payload:
            raise RuntimeError(f"{arm} deterministic audit payloads differ.")
        arms[arm] = {
            "groups": expected_groups,
            "canonical_sha256": canonical_sha256(left_payload),
            "exact_match": True,
        }
    report = {
        "schema_version": SCHEMA_VERSION,
        "passed": True,
        "expected_groups": expected_groups,
        "right_prefix": right_prefix,
        "left_root": str(left_root),
        "right_root": str(right_root),
        "arms": arms,
    }
    if output_path is not None:
        _write_or_verify_json(Path(output_path), report)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run = subparsers.add_parser("run", help="Run one isolated temperature arm.")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--temperature", type=float, required=True)
    run.add_argument("--output-root", type=Path)
    run.add_argument("--max-groups", type=int, default=512)
    run.add_argument("--device", default="cuda:0")

    compare = subparsers.add_parser("compare", help="Build paired A/B comparison.")
    compare.add_argument("--config", type=Path, required=True)
    compare.add_argument("--output-root", type=Path)
    compare.add_argument("--expected-groups", type=int, default=512)

    verify = subparsers.add_parser("verify", help="Independently verify all outputs.")
    verify.add_argument("--config", type=Path, required=True)
    verify.add_argument("--output-root", type=Path)
    verify.add_argument("--expected-groups", type=int, default=512)
    verify.add_argument("--smoke-root", type=Path)
    verify.add_argument("--require-determinism", action="store_true")

    deterministic = subparsers.add_parser(
        "determinism", help="Require two runs to have identical raw semantic audits."
    )
    deterministic.add_argument("--left-root", type=Path, required=True)
    deterministic.add_argument("--right-root", type=Path, required=True)
    deterministic.add_argument("--expected-groups", type=int, required=True)
    deterministic.add_argument("--right-prefix", action="store_true")
    deterministic.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "run":
        result = run_arm(
            args.config,
            temperature=args.temperature,
            output_root=args.output_root,
            max_groups=args.max_groups,
            device=args.device,
        )
    elif args.command == "compare":
        result = compare_outputs(
            args.config,
            output_root=args.output_root,
            expected_groups=args.expected_groups,
        )
    elif args.command == "verify":
        result = verify_outputs(
            args.config,
            output_root=args.output_root,
            expected_groups=args.expected_groups,
            smoke_root=args.smoke_root,
            require_determinism=args.require_determinism,
        )
    elif args.command == "determinism":
        result = verify_determinism(
            left_root=args.left_root,
            right_root=args.right_root,
            expected_groups=args.expected_groups,
            right_prefix=args.right_prefix,
            output_path=args.output,
        )
    else:  # pragma: no cover - argparse enforces a known command
        raise AssertionError(args.command)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "aggregate_audit",
    "build_comparison",
    "compare_outputs",
    "load_temperature_config",
    "main",
    "parameter_fingerprint",
    "run_arm",
    "verify_determinism",
    "verify_outputs",
]
