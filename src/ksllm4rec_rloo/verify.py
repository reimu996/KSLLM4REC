"""Independent final verification of gates, training audit, recovery, and probes."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

from ksllm4rec_grpo.data import iter_groups
from ksllm4rec_grpo.trie import SidPrefixTrie
from ksllm4rec_orpo.data import Sid

from . import contract
from .checkpoint import load_latest_recovery
from .gates import load_training_gates
from .integrity import file_record, sha256_file
from .modeling import validate_adapter_contract
from .objective import ObjectiveBranch, rewards_and_advantages
from .probe import probe_run_signature
from .trainer import learning_rate_for_window


def _load_json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object: {path}")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Corrupt JSONL line {line_number}: {path}") from exc
            if not isinstance(value, dict):
                raise RuntimeError(f"JSONL line {line_number} is not an object: {path}")
            rows.append(value)
    return rows


def _close(left: float, right: float, tolerance: float = 1.0e-6) -> bool:
    return math.isclose(float(left), float(right), rel_tol=tolerance, abs_tol=tolerance)


def _adapter_record(path: Path) -> dict[str, Any]:
    expected = {"adapter_model.safetensors", "adapter_config.json"}
    missing = [name for name in expected if not (path / name).is_file()]
    if missing:
        raise RuntimeError(f"Adapter is incomplete at {path}: {missing}")
    adapter_contract = validate_adapter_contract(path)
    if adapter_contract.source_dropout != 0.0:
        raise RuntimeError(f"Saved adapter differs from r64/alpha64/dropout0: {path}")
    return {
        "adapter_model.safetensors": file_record(path / "adapter_model.safetensors"),
        "adapter_config.json": file_record(path / "adapter_config.json"),
    }


def _verify_probe(
    config: dict[str, Any],
    *,
    adapter_dir: Path,
    trie: SidPrefixTrie,
    trie_dir: Path,
    groups_path: Path,
    calibration_ids_path: Path,
    config_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    expected_signature = probe_run_signature(
        config,
        adapter_dir,
        trie_dir,
        groups_path=groups_path,
        calibration_ids_path=calibration_ids_path,
        config_path=config_path,
    )
    if _load_json(output_dir / "probe_state.json") != expected_signature:
        raise RuntimeError(f"Probe state is not bound to {adapter_dir}.")
    report = _load_json(output_dir / "probe_report.json")
    if report.get("run_signature") != expected_signature:
        raise RuntimeError(f"Probe report is not bound to {adapter_dir}.")
    source = _load_jsonl(Path(config["evaluation"]["fixed_probe"]))
    predictions = _load_jsonl(output_dir / "predictions.jsonl")
    if len(source) != 1024 or len(predictions) != 1024:
        raise RuntimeError("Probe must contain exactly 1,024 completed predictions.")
    metrics: dict[str, Counter[str]] = {
        "text_to_sid": Counter(),
        "recommend": Counter(),
    }
    for index, (target_row, prediction) in enumerate(
        zip(source, predictions, strict=True)
    ):
        target = Sid.parse(target_row["target_sid"])
        predicted = Sid.parse(prediction["predicted_sid"])
        reachable = trie.contains(target)
        valid = trie.contains(predicted)
        exact = target == predicted
        expected = {
            "probe_index": index,
            "run_signature": expected_signature["sha256"],
            "task": target_row["task"],
            "source_row_sha256": target_row["source_row_sha256"],
            "target_sid": target.render(),
            "target_reachable": reachable,
            "predicted_sid": predicted.render(),
            "valid_sid": valid,
            "exact": exact,
        }
        for key, value in expected.items():
            if prediction.get(key) != value:
                raise RuntimeError(f"Probe prediction {index} differs at {key}.")
        bucket = metrics[target_row["task"]]
        bucket["rows"] += 1
        bucket["reachable_targets"] += int(reachable)
        bucket["valid_predictions"] += int(valid)
        bucket["exact"] += int(exact)
        bucket["exact_reachable"] += int(exact and reachable)
    if report.get("rows") != 1024 or report.get("beam_size") != 16:
        raise RuntimeError("Probe report is incomplete or not beam-16.")
    if {name: dict(value) for name, value in metrics.items()} != report.get("metrics"):
        raise RuntimeError("Probe aggregate metrics do not match predictions.")
    return report


def verify_run(
    config: dict[str, Any],
    *,
    config_path: Path,
    groups_path: Path,
    trie_dir: Path,
    calibration_ids_path: Path,
    run_dir: Path,
    probe_root: Path,
    gate_root: Path,
) -> dict[str, Any]:
    gates = load_training_gates(
        config=config,
        groups_path=groups_path,
        trie_dir=trie_dir,
        calibration_ids_path=calibration_ids_path,
        structure_path=gate_root / "structure_gate.json",
        calibration_path=gate_root / "calibration_gate.json",
        probability_path=gate_root / "probability_gate.json",
        memory_path=gate_root / "memory_gate.json",
        signal_path=gate_root / "signal_gate.json",
        timing_path=gate_root / "timing_gate.json",
        config_path=config_path,
    )
    groups = list(iter_groups(groups_path))
    trie = SidPrefixTrie.load(
        trie_dir, expected_leaf_count=contract.EXPECTED_TRIE_LEAVES
    )
    adapters = {
        "epoch_001": _adapter_record(run_dir / "epoch_001"),
        "epoch_002": _adapter_record(run_dir / "epoch_002"),
    }
    hashes = [
        adapters[name]["adapter_model.safetensors"]["sha256"]
        for name in ("epoch_001", "epoch_002")
    ]
    if len(set(hashes)) != 2 or contract.SFT_ADAPTER_SHA256 in hashes:
        raise RuntimeError("Epoch adapters are not two distinct trained checkpoints.")

    summary = _load_json(run_dir / "run_summary.json")
    state = summary.get("state")
    expected_state = {
        "epoch_index": 3,
        "next_group_offset": 0,
        "window_step": 4_254,
        "groups_completed": 34_032,
        "rollout_chunk": 8,
        "loss_chunk": 8,
        "lambda0": gates.lambda0,
    }
    if not isinstance(state, dict):
        raise RuntimeError("Final summary has no state mapping.")
    for key, value in expected_state.items():
        if state.get(key) != value:
            raise RuntimeError(f"Final state differs at {key}: {state.get(key)} != {value}")
    optimizer_updates = int(state.get("optimizer_update_step", -1))
    if not 0 < optimizer_updates <= 4_254:
        raise RuntimeError("Final optimizer update count is invalid.")
    if (
        summary.get("runtime_signature") != gates.runtime_signature
        or summary.get("resolved_contract") != gates.resolved_contract
        or summary.get("completed") is not True
    ):
        raise RuntimeError("Final summary is not bound to the accepted gates.")

    recovery = load_latest_recovery(
        run_dir / "recovery", gates.runtime_signature, gates.resolved_contract
    )
    if recovery is None:
        raise RuntimeError("Final recovery checkpoint is missing.")
    _, recovery_cursor, _ = recovery
    if recovery_cursor.__dict__ != state:
        raise RuntimeError("Final recovery cursor and run summary differ.")

    audit = _load_jsonl(run_dir / "train_audit.jsonl")
    progress = _load_jsonl(run_dir / "train_progress.jsonl")
    if len(audit) != 34_032 or len(progress) != 4_254:
        raise RuntimeError("Training audit/progress row counts are incomplete.")
    branch_counts: Counter[str] = Counter()
    tier_counts: Counter[str] = Counter()
    reward_means: list[float] = []
    previous_update = 0
    for window_index, row in enumerate(progress):
        window_step = window_index + 1
        if row.get("window_step") != window_step:
            raise RuntimeError(f"Progress cursor differs at window {window_step}.")
        expected_lr = learning_rate_for_window(config, window_index)
        if not _close(row.get("learning_rate"), expected_lr, tolerance=1.0e-10):
            raise RuntimeError(f"Learning rate differs at window {window_step}.")
        branches = row.get("branch_counts", {})
        if sum(int(value) for value in branches.values()) != 8:
            raise RuntimeError(f"Window {window_step} does not contain eight groups.")
        updated = any(
            int(branches.get(name, 0)) > 0
            for name in (
                ObjectiveBranch.RLOO.value,
                ObjectiveBranch.GT_SET_ANCHOR.value,
            )
        )
        current_update = int(row.get("optimizer_update_step", -1))
        expected_update = previous_update + int(updated)
        if current_update != expected_update or row.get("optimizer_updated") != updated:
            raise RuntimeError(f"Optimizer counter differs at window {window_step}.")
        previous_update = current_update
    if previous_update != optimizer_updates:
        raise RuntimeError("Progress and final optimizer counts differ.")

    for index, row in enumerate(audit):
        epoch = index // len(groups) + 1
        group = groups[index % len(groups)]
        line_number = index + 1
        if (
            row.get("epoch") != epoch
            or row.get("group_id") != group.group_id
            or row.get("source_lines") != list(group.source_lines)
            or row.get("positive_sids") != list(group.positive_sids)
            or row.get("candidate_indices") != list(range(16))
            or row.get("gt_injection_count") != 0
            or row.get("divide_by_std") is not False
        ):
            raise RuntimeError(f"Audit/input binding failed at row {line_number}.")
        candidates = tuple(Sid.parse(value) for value in row["candidate_sids"])
        positives = tuple(Sid.parse(value) for value in group.positive_sids)
        if len(candidates) != 16 or not all(trie.contains(sid) for sid in candidates):
            raise RuntimeError(f"Illegal G=16 candidates at row {line_number}.")
        expected = rewards_and_advantages(candidates, positives)
        if (
            row.get("reward_tiers") != list(expected.tiers)
            or row.get("branch") != expected.branch.value
            or not _close(row.get("reward_mean_live"), expected.mean_reward.item())
            or not _close(row.get("reward_mean_final"), expected.mean_reward.item())
            or any(
                not _close(actual, target)
                for actual, target in zip(
                    row.get("rewards", []), expected.rewards.tolist(), strict=True
                )
            )
            or any(
                not _close(actual, target)
                for actual, target in zip(
                    row.get("advantages", []),
                    expected.advantages.tolist(),
                    strict=True,
                )
            )
        ):
            raise RuntimeError(f"Reward/RLOO branch differs at row {line_number}.")
        delta = float(row.get("max_sample_replay_logp_difference", float("inf")))
        if not math.isfinite(delta) or delta > float(
            config["gates"]["max_logp_difference"]
        ):
            raise RuntimeError(f"Probability alignment failed at row {line_number}.")
        if expected.branch == ObjectiveBranch.SKIP:
            if row.get("loss") is not None or row.get("decision_tokens") != 0:
                raise RuntimeError(f"Skip branch has a loss at row {line_number}.")
        else:
            if not math.isfinite(float(row.get("loss"))):
                raise RuntimeError(f"Active branch has non-finite loss at row {line_number}.")
        branch_counts[expected.branch.value] += 1
        tier_counts.update(expected.tiers)
        reward_means.append(float(expected.mean_reward))

    metrics = summary.get("metrics", {})
    if (
        metrics.get("groups") != len(audit)
        or metrics.get("branch_counts") != dict(branch_counts)
        or metrics.get("reward_tier_counts") != dict(tier_counts)
        or metrics.get("gt_injection_count") != 0
        or not _close(metrics.get("reward_mean"), fmean(reward_means))
    ):
        raise RuntimeError("Final aggregate metrics do not equal the audit.")

    probes = {
        "epoch_000": _verify_probe(
            config,
            adapter_dir=Path(config["model"]["sft_adapter"]),
            trie=trie,
            trie_dir=trie_dir,
            groups_path=groups_path,
            calibration_ids_path=calibration_ids_path,
            config_path=config_path,
            output_dir=probe_root / "epoch_000",
        ),
        "epoch_001": _verify_probe(
            config,
            adapter_dir=run_dir / "epoch_001",
            trie=trie,
            trie_dir=trie_dir,
            groups_path=groups_path,
            calibration_ids_path=calibration_ids_path,
            config_path=config_path,
            output_dir=probe_root / "epoch_001",
        ),
        "epoch_002": _verify_probe(
            config,
            adapter_dir=run_dir / "epoch_002",
            trie=trie,
            trie_dir=trie_dir,
            groups_path=groups_path,
            calibration_ids_path=calibration_ids_path,
            config_path=config_path,
            output_dir=probe_root / "epoch_002",
        ),
    }
    return {
        "schema_version": 1,
        "spec_version": config["spec_version"],
        "passed": True,
        "runtime_signature": gates.runtime_signature,
        "resolved_contract": gates.resolved_contract,
        "adapters": adapters,
        "training": {
            "groups": len(audit),
            "windows": len(progress),
            "optimizer_updates": optimizer_updates,
            "branch_counts": dict(branch_counts),
            "reward_tier_counts": dict(tier_counts),
            "reward_mean": fmean(reward_means),
            "gt_injection_count": 0,
        },
        "probes": probes,
        "gate_report_sha256": {
            name: sha256_file(gate_root / f"{name}_gate.json")
            for name in (
                "structure",
                "calibration",
                "probability",
                "memory",
                "signal",
                "timing",
            )
        },
    }


__all__ = ["verify_run"]
