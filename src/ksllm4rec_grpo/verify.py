"""Strict end-to-end verification for completed GRPO Spec V3.1."""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from statistics import fmean
from typing import Any

from ksllm4rec_orpo.data import Sid

from .contract import (
    POSITIVE_SET_SIZE_DISTRIBUTION,
    profile_for_config,
)
from .data import RecommendationGroup, iter_groups
from .fingerprint import runtime_signature
from .integrity import sha256_file
from .objective import (
    rewards_and_advantages,
    select_forced_gt,
    should_add_gt,
)
from .probe import probe_run_signature
from .trie import SidPrefixTrie
from .trainer import training_schedule


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise RuntimeError(f"Invalid JSONL at {path}:{line_number}.") from exc
    return rows


def _close(actual: float, expected: float, tolerance: float = 1e-6) -> bool:
    return math.isclose(
        float(actual), float(expected), rel_tol=tolerance, abs_tol=tolerance
    )


def _require_adapter(path: Path) -> dict[str, Any]:
    files = {}
    for name in ("adapter_model.safetensors", "adapter_config.json"):
        item = path / name
        if not item.is_file() or item.stat().st_size == 0:
            raise RuntimeError(f"Incomplete epoch adapter: {item}")
        files[name] = {"size": item.stat().st_size, "sha256": sha256_file(item)}
    config = _load_json(path / "adapter_config.json")
    expected = {
        "r": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.0,
        "bias": "none",
        "task_type": "CAUSAL_LM",
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise RuntimeError(f"Saved adapter config differs: {path}: {mismatches}")
    return files


def _verify_inputs(
    config: dict[str, Any], groups_path: Path, trie_dir: Path
) -> tuple[list[RecommendationGroup], SidPrefixTrie, dict[str, Any]]:
    profile = profile_for_config(config)
    manifest = _load_json(groups_path.parent / "data_manifest.json")
    source_record = manifest.get("source") or manifest.get("source_lock", {}).get(
        "source"
    )
    if (
        not isinstance(source_record, dict)
        or source_record.get("sha256") != profile.source_sha256
        or sha256_file(groups_path) != profile.groups_sha256
        or manifest["groups"]["sha256"] != profile.groups_sha256
        or manifest["groups"]["rows"] != profile.groups
        or manifest["positive_rows"] != profile.positives
        or manifest["unique_positive_sids"] != profile.unique_positive_sids
    ):
        raise RuntimeError("Grouped-data manifest differs from the frozen input.")
    groups = list(iter_groups(groups_path))
    positive_rows = sum(len(group.positive_sids) for group in groups)
    unique_positives = len({sid for group in groups for sid in group.positive_sids})
    distribution = dict(Counter(len(group.positive_sids) for group in groups))
    if (
        len(groups) != profile.groups
        or positive_rows != profile.positives
        or unique_positives != profile.unique_positive_sids
        or (
            profile.provenance is None
            and distribution != POSITIVE_SET_SIZE_DISTRIBUTION
        )
    ):
        raise RuntimeError("Grouped data content differs from the frozen contract.")
    if sha256_file(trie_dir / "manifest.json") != profile.trie_manifest_sha256:
        raise RuntimeError("Trie manifest hash differs from the frozen contract.")
    trie = SidPrefixTrie.load(
        trie_dir, expected_leaf_count=profile.unique_sids
    )
    if (
        trie.metadata.get("source_sha256") != profile.source_sha256
        or trie.metadata.get("strategy") != profile.trie_strategy
        or trie.counts["a_nodes"] != profile.domain_a_nodes
        or trie.counts["ab_nodes"] != profile.domain_ab_nodes
        or {
            domain: trie.counts["by_domain"][domain]["leaves"]
            for domain in profile.domain_sids
        }
        != dict(profile.domain_sids)
    ):
        raise RuntimeError("Configured SID trie differs from the frozen contract.")
    return groups, trie, runtime_signature(config, groups_path, trie_dir)


def _verify_gates(
    config: dict[str, Any], gate_root: Path, signature: dict[str, Any]
) -> dict[str, Any]:
    reports = {
        "memory": _load_json(gate_root / "memory_gate.json"),
        "signal": _load_json(gate_root / "signal_gate.json"),
        "timing": _load_json(gate_root / "timing_gate.json"),
    }
    for name, report in reports.items():
        if report.get("runtime_signature") != signature:
            raise RuntimeError(f"{name} gate belongs to a different runtime contract.")
    memory = reports["memory"]
    if memory["rollout_chunk"] != memory["loss_chunk"] or not any(
        attempt.get("success")
        and attempt.get("phase") == "loss"
        and attempt.get("chunk") == memory["loss_chunk"]
        and attempt.get("peak_reserved_gib", math.inf)
        <= float(config["memory"]["max_reserved_gib"])
        for attempt in memory["attempts"]
    ):
        raise RuntimeError("Unified memory gate did not pass.")
    signal = reports["signal"]
    signal_state = signal.get("state", {})
    if (
        signal.get("passed") is not True
        or signal.get("groups_run") != int(config["gates"]["signal_groups"])
        or signal.get("signal_groups", 0) <= 0
        or signal.get("gradient_norm_max", 0.0) <= 0.0
        or signal.get("parameter_max_change_this_invocation", 0.0) <= 0.0
        or signal_state.get("rollout_chunk") != memory["rollout_chunk"]
        or signal_state.get("loss_chunk") != memory["loss_chunk"]
    ):
        raise RuntimeError("Signal gate did not prove a finite policy update.")
    timing = reports["timing"]
    timing_state = timing.get("state", {})
    if (
        timing.get("passed") is not True
        or timing.get("groups_run") != int(config["gates"]["timing_groups"])
        or timing.get("projected_two_epoch_hours", math.inf)
        > float(config["gates"]["max_projected_hours"])
        or timing_state.get("rollout_chunk") != memory["rollout_chunk"]
        or timing_state.get("loss_chunk") != memory["loss_chunk"]
    ):
        raise RuntimeError("Timing gate did not pass.")
    return reports


def _verify_candidate_branch(
    row: dict[str, Any],
    group: RecommendationGroup,
    epoch: int,
    live: tuple[Sid, ...],
    final: tuple[Sid, ...],
    positives: tuple[Sid, ...],
    line_number: int,
) -> None:
    has_live_gt = any(sid in positives for sid in live)
    if bool(row["has_live_gt"]) != has_live_gt:
        raise RuntimeError(f"Live GT flag mismatch at audit line {line_number}.")
    if has_live_gt:
        expected_add = False
        expected_uniform = None
    else:
        expected_add, expected_uniform = should_add_gt(
            group_id=group.group_id, epoch_index=epoch
        )
    if bool(row["add_gt"]) != expected_add:
        raise RuntimeError(f"add_gt decision mismatch at audit line {line_number}.")
    if expected_uniform is None:
        if row["add_gt_uniform"] is not None:
            raise RuntimeError(f"Unexpected add_gt hash at audit line {line_number}.")
    elif not _close(row["add_gt_uniform"], expected_uniform, tolerance=0.0):
        raise RuntimeError(f"add_gt hash mismatch at audit line {line_number}.")

    if expected_add:
        expected_gt = select_forced_gt(
            positives, group_id=group.group_id, epoch_index=epoch
        )
        if (
            final[:7] != live[:7]
            or final[7] != expected_gt
            or row["forced_mask"] != [False] * 7 + [True]
            or row["forced_gt"] != expected_gt.render()
            or row["discarded_rollout"] != live[7].render()
        ):
            raise RuntimeError(f"Forced-GT replacement mismatch at line {line_number}.")
    elif (
        final != live
        or row["forced_mask"] != [False] * 8
        or row["forced_gt"] is not None
        or row["discarded_rollout"] is not None
    ):
        raise RuntimeError(f"Non-forced candidates changed at line {line_number}.")


def _verify_objective(
    row: dict[str, Any],
    final: tuple[Sid, ...],
    positives: tuple[Sid, ...],
    line_number: int,
    beta: float,
) -> None:
    expected = rewards_and_advantages(final, positives)
    pairs = (
        (row["rewards"], expected.rewards.tolist(), "rewards"),
        (row["advantages"], expected.advantages.tolist(), "advantages"),
    )
    for actual_values, expected_values, label in pairs:
        if len(actual_values) != 8 or any(
            not _close(actual, target)
            for actual, target in zip(actual_values, expected_values, strict=True)
        ):
            raise RuntimeError(f"{label} mismatch at audit line {line_number}.")
    if (
        not _close(row["reward_mean"], expected.mean_reward.item())
        or not _close(row["reward_std"], expected.std_reward.item())
        or row["reward_tiers"] != list(expected.tiers)
    ):
        raise RuntimeError(f"Reward statistics mismatch at line {line_number}.")
    if expected.std_reward.item() == 0.0 and any(row["advantages"]):
        raise RuntimeError(
            f"Equal rewards produced nonzero advantage at line {line_number}."
        )
    old_new_delta = float(row["max_old_new_logp_delta"])
    generation_delta = float(row["max_generation_rescore_logp_delta"])
    if not math.isfinite(old_new_delta) or old_new_delta > 1.0e-5:
        raise RuntimeError(f"Old/current policy delta failed at line {line_number}.")
    if not math.isfinite(generation_delta) or generation_delta < 0.0:
        raise RuntimeError(f"Invalid generation diagnostic at line {line_number}.")

    cursor = 0
    reconstructed_group_loss = 0.0
    for chunk in row["chunks"]:
        start, stop = int(chunk["start"]), int(chunk["stop"])
        if start != cursor or not start < stop <= 8:
            raise RuntimeError(f"Loss chunks do not cover G=8 at line {line_number}.")
        width = stop - start
        losses = chunk["per_candidate_loss"]
        kls = chunk["per_candidate_reference_kl"]
        counts = chunk["decision_counts"]
        if not (len(losses) == len(kls) == len(counts) == width) or any(
            int(count) <= 0 for count in counts
        ):
            raise RuntimeError(f"Invalid candidate loss shape at line {line_number}.")
        for index, (actual_loss, kl) in enumerate(zip(losses, kls, strict=True)):
            expected_loss = -float(row["advantages"][start + index]) + beta * float(kl)
            if (
                not math.isfinite(float(actual_loss))
                or float(kl) < -1e-7
                or not _close(actual_loss, expected_loss, tolerance=2e-5)
            ):
                raise RuntimeError(
                    f"Per-candidate GRPO loss mismatch at line {line_number}."
                )
        if not _close(chunk["loss"], fmean(float(value) for value in losses)):
            raise RuntimeError(f"Chunk loss reduction mismatch at line {line_number}.")
        reconstructed_group_loss += float(chunk["loss"]) * width / 8.0
        cursor = stop
    if cursor != 8 or not _close(row["loss"], reconstructed_group_loss):
        raise RuntimeError(f"Group loss reduction mismatch at line {line_number}.")


def _verify_probe(
    config: dict[str, Any],
    trie: SidPrefixTrie,
    trie_dir: Path,
    adapter_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    profile = profile_for_config(config)
    expected_signature = probe_run_signature(config, adapter_dir, trie_dir)
    if _load_json(output_dir / "probe_state.json") != expected_signature:
        raise RuntimeError(f"Probe state is not bound to {adapter_dir}.")
    report = _load_json(output_dir / "probe_report.json")
    if report.get("run_signature") != expected_signature:
        raise RuntimeError(f"Probe report is not bound to {adapter_dir}.")
    source_rows = _load_jsonl(Path(config["evaluation"]["fixed_probe"]))
    if (
        sha256_file(Path(config["evaluation"]["fixed_probe"]))
        != profile.fixed_probe_sha256
    ):
        raise RuntimeError("Fixed probe input hash changed.")
    predictions = _load_jsonl(output_dir / "predictions.jsonl")
    if len(source_rows) != 1024 or len(predictions) != 1024:
        raise RuntimeError("Probe predictions are incomplete.")
    metrics: dict[str, Counter[str]] = {
        "text_to_sid": Counter(),
        "recommend": Counter(),
    }
    for index, (source, prediction) in enumerate(
        zip(source_rows, predictions, strict=True)
    ):
        target = Sid.parse(source["target_sid"])
        predicted = Sid.parse(prediction["predicted_sid"])
        reachable = trie.contains(target)
        valid = trie.contains(predicted)
        exact = predicted == target
        if (
            prediction["probe_index"] != index
            or prediction["run_signature"] != expected_signature["sha256"]
            or prediction["task"] != source["task"]
            or prediction["source_row_sha256"] != source["source_row_sha256"]
            or prediction["target_sid"] != target.render()
            or prediction["target_reachable"] != reachable
            or prediction["valid_sid"] != valid
            or prediction["exact"] != exact
        ):
            raise RuntimeError(f"Probe prediction mismatch at index {index}.")
        bucket = metrics[source["task"]]
        bucket["rows"] += 1
        bucket["reachable_targets"] += int(reachable)
        bucket["valid_predictions"] += int(valid)
        bucket["exact"] += int(exact)
        bucket["exact_reachable"] += int(exact and reachable)
    for task, expected_reachable in dict(profile.probe_reachable or {}).items():
        if metrics[task]["reachable_targets"] != expected_reachable:
            raise RuntimeError(f"Probe reachability changed for {task}.")
        if dict(metrics[task]) != report["metrics"][task]:
            raise RuntimeError(f"Probe metrics mismatch for {task}.")
    if report["rows"] != 1024 or report["beam_size"] != 16:
        raise RuntimeError("Probe report configuration differs from the contract.")
    return report


def verify_run(
    config: dict[str, Any],
    *,
    groups_path: Path,
    trie_dir: Path,
    run_dir: Path,
    probe_root: Path,
    gate_root: Path,
) -> dict[str, Any]:
    profile = profile_for_config(config)
    groups, trie, signature = _verify_inputs(config, groups_path, trie_dir)
    gates = _verify_gates(config, gate_root, signature)
    group_count = len(groups)
    expected_total_groups = group_count * int(config["train"]["epochs"])

    epochs = int(config["train"]["epochs"])
    adapters = {
        f"epoch_{epoch:03d}": _require_adapter(run_dir / f"epoch_{epoch:03d}")
        for epoch in range(1, epochs + 1)
    }
    adapter_hashes = [
        adapters[key]["adapter_model.safetensors"]["sha256"]
        for key in sorted(adapters)
    ]
    if len(set(adapter_hashes)) != epochs or profile.adapter_sha256 in adapter_hashes:
        raise RuntimeError("Epoch adapters did not change independently from SFT.")

    summary = _load_json(run_dir / "run_summary.json")
    summary_state = summary.get("state")
    if not isinstance(summary_state, dict):
        raise RuntimeError("Final training summary has no runtime state.")
    initial_chunk = int(gates["memory"]["rollout_chunk"])
    candidates = [int(value) for value in config["memory"]["chunk_candidates"]]
    permitted_chunks = candidates[candidates.index(initial_chunk) :]
    final_rollout_chunk = int(summary_state.get("rollout_chunk", -1))
    final_loss_chunk = int(summary_state.get("loss_chunk", -1))
    if (
        final_rollout_chunk != final_loss_chunk
        or final_rollout_chunk not in permitted_chunks
    ):
        raise RuntimeError("Final chunk is not an approved unified OOM fallback.")
    total_steps, _ = training_schedule(config, group_count)
    expected_state = {
        "epoch_index": epochs + 1,
        "next_group_offset": 0,
        "global_step": total_steps,
        "groups_completed": expected_total_groups,
        "rollout_chunk": final_rollout_chunk,
        "loss_chunk": final_loss_chunk,
    }
    if (
        summary.get("state") != expected_state
        or summary.get("groups_run") != expected_total_groups
    ):
        raise RuntimeError(f"Unexpected final training summary: {summary.get('state')}")

    latest = _load_json(run_dir / "recovery/latest.json")
    recovery_manifest = _load_json(
        run_dir / "recovery" / latest["checkpoint"] / "manifest.json"
    )
    if (
        latest["global_step"] != total_steps
        or recovery_manifest["contract_signature"] != signature
        or recovery_manifest["cursor"] != expected_state
    ):
        raise RuntimeError("Final recovery point is not bound to the completed run.")

    audit = _load_jsonl(run_dir / "train_audit.jsonl")
    progress = _load_jsonl(run_dir / "train_progress.jsonl")
    if len(audit) != expected_total_groups or len(progress) != total_steps:
        raise RuntimeError("Training audit/progress row counts are incomplete.")
    previous_chunk = initial_chunk
    for index, row in enumerate(progress, start=1):
        rollout_chunk = int(row.get("rollout_chunk", -1))
        loss_chunk = int(row.get("loss_chunk", -1))
        if (
            rollout_chunk != loss_chunk
            or rollout_chunk not in permitted_chunks
            or rollout_chunk > previous_chunk
        ):
            raise RuntimeError(f"Invalid chunk transition at optimizer step {index}.")
        previous_chunk = rollout_chunk
    if previous_chunk != final_rollout_chunk:
        raise RuntimeError("Progress and final chunk states differ.")
    for retry in summary.get("oom_retries_this_invocation", []):
        if int(retry["new_chunk"]) * 2 != int(retry["previous_chunk"]):
            raise RuntimeError("OOM retry did not halve the unified chunk.")
    step_sizes = Counter()
    audit_counts = Counter()
    maximum_delta = 0.0
    maximum_generation_delta = 0.0
    losses = []
    beta = float(config["loss"]["reference_beta"])
    for index, row in enumerate(audit):
        line_number = index + 1
        epoch = index // group_count + 1
        group = groups[index % group_count]
        if (
            row["epoch"] != epoch
            or row["group_id"] != group.group_id
            or row["positive_sids"] != list(group.positive_sids)
            or row["source_lines"] != list(group.source_lines)
        ):
            raise RuntimeError(f"Audit/input binding mismatch at line {line_number}.")
        live = tuple(Sid.parse(value) for value in row["live_sids"])
        final = tuple(Sid.parse(value) for value in row["final_sids"])
        positives = tuple(Sid.parse(value) for value in group.positive_sids)
        if (
            len(live) != 8
            or len(final) != 8
            or not all(trie.contains(sid) for sid in (*live, *final))
        ):
            raise RuntimeError(f"Illegal G=8 candidates at line {line_number}.")
        _verify_candidate_branch(row, group, epoch, live, final, positives, line_number)
        _verify_objective(row, final, positives, line_number, beta)
        maximum_delta = max(
            maximum_delta,
            float(row["max_old_new_logp_delta"]),
        )
        maximum_generation_delta = max(
            maximum_generation_delta,
            float(row["max_generation_rescore_logp_delta"]),
        )
        losses.append(float(row["loss"]))
        step_sizes[int(row["optimizer_step"])] += 1
        audit_counts["groups"] += 1
        audit_counts["forced"] += int(row["add_gt"])
        audit_counts["live_exact"] += int(row["has_live_gt"])
        audit_counts["signal"] += int(any(row["advantages"]))
        audit_counts[f"epoch_{epoch}"] += 1
    accumulation = int(config["train"]["gradient_accumulation_groups"])
    steps_per_epoch = math.ceil(group_count / accumulation)
    tail = group_count % accumulation
    for step in range(1, total_steps + 1):
        final_step_in_epoch = step % steps_per_epoch == 0
        expected = tail if final_step_in_epoch and tail else accumulation
        if step_sizes[step] != expected or progress[step - 1]["optimizer_step"] != step:
            raise RuntimeError(f"Optimizer step {step} has an invalid group reduction.")
    if (
        progress[-1]["groups_completed"] != expected_total_groups
        or summary["forced_groups"] != audit_counts["forced"]
        or summary["signal_groups"] != audit_counts["signal"]
        or summary["exact_live_groups"] != audit_counts["live_exact"]
        or not _close(summary["loss_mean"], fmean(losses))
    ):
        raise RuntimeError("Training summary does not equal the full audit.")

    probes = {
        f"epoch_{epoch:03d}": _verify_probe(
            config,
            trie,
            trie_dir,
            run_dir / f"epoch_{epoch:03d}",
            probe_root / f"epoch_{epoch:03d}",
        )
        for epoch in range(1, epochs + 1)
    }
    return {
        "spec_version": config["spec_version"],
        "passed": True,
        "runtime_signature": signature,
        "adapters": adapters,
        "gates": gates,
        "training_summary": summary,
        "audit_counts": dict(audit_counts),
        "maximum_on_policy_logp_delta": maximum_delta,
        "maximum_generation_rescore_logp_delta": maximum_generation_delta,
        "optimizer_steps": len(step_sizes),
        "probes": probes,
    }
