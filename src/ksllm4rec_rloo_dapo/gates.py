"""Measured preflight gates bound to the exact RLOO-DAPO runtime signature."""

from __future__ import annotations

from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import torch

from ksllm4rec_grpo.constraint import RecommendationGrammar
from ksllm4rec_grpo.prompt import encode_prompt
from ksllm4rec_orpo.data import Sid
from ksllm4rec_rloo.integrity import canonical_sha256, sha256_file
from ksllm4rec_rloo.modeling import load_policy_model
from ksllm4rec_rloo.rollout import rollout_group

from . import contract
from .fingerprint import validate_runtime_signature
from .objective import clipped_rloo_token_sum
from .rollout import (
    PromptRequest,
    canonicalize_prompt_rollout,
    rollout_prompt_batch,
)
from .scoring import grammar_completion_width, score_completions_dense_fixed
from .trainer import configure_deterministic_runtime, load_groups_and_trie, run_training


GATE_SCHEMA_VERSION = 1


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                dict(value),
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


def _bound_report(
    gate: str,
    signature: Mapping[str, Any],
    *,
    passed: bool,
    **values: Any,
) -> dict[str, Any]:
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "gate": gate,
        "passed": bool(passed),
        "runtime_signature_sha256": signature["sha256"],
        **values,
    }


def expected_input_sha256() -> dict[str, str]:
    return {
        "base_model": contract.BASE_MODEL_SHA256,
        "sft_adapter": contract.SFT_ADAPTER_SHA256,
        "sft_adapter_config": contract.SFT_CONFIG_SHA256,
        "tokenizer": contract.TOKENIZER_SHA256,
        "source": contract.SOURCE_DATA_SHA256,
        "provenance": contract.PROVENANCE_SHA256,
        "groups": contract.GROUPS_SHA256,
        "trie_manifest": contract.TRIE_MANIFEST_SHA256,
        "fixed_probe": contract.FIXED_PROBE_SHA256,
    }


def collect_input_sha256(config: Mapping[str, Any]) -> dict[str, str]:
    base = Path(config["model"]["base_model"])
    adapter = Path(config["model"]["sft_adapter"])
    tokenizer = Path(config["model"]["tokenizer"])
    return {
        "base_model": sha256_file(base / "model.safetensors"),
        "sft_adapter": sha256_file(adapter / "adapter_model.safetensors"),
        "sft_adapter_config": sha256_file(adapter / "adapter_config.json"),
        "tokenizer": sha256_file(tokenizer / "tokenizer.json"),
        "source": sha256_file(Path(config["data"]["source"])),
        "provenance": sha256_file(Path(config["data"]["provenance"])),
        "groups": sha256_file(Path(config["output"]["groups_dir"]) / "groups.jsonl"),
        "trie_manifest": sha256_file(
            Path(config["output"]["trie_dir"]) / "manifest.json"
        ),
        "fixed_probe": sha256_file(Path(config["evaluation"]["fixed_probe"])),
    }


def structure_gate(
    config: dict[str, Any],
    signature: dict[str, Any],
    *,
    output: Path,
) -> dict[str, Any]:
    validate_runtime_signature(signature)
    groups, trie = load_groups_and_trie(config)
    positive_edges = sum(len(group.positive_sids) for group in groups)
    group_ids = [group.group_id for group in groups]
    input_sha = collect_input_sha256(config)
    expected_sha = expected_input_sha256()
    structure = {
        "groups": len(groups),
        "positive_edges": positive_edges,
        "unique_group_ids": len(set(group_ids)),
        "trie_leaves": trie.leaf_count,
        "trie_a_nodes": int(trie.a_values.size),
        "trie_ab_nodes": int(trie.b_values.size),
        "effective_groups_per_window": contract.EFFECTIVE_GROUPS_PER_WINDOW,
        "group_size": contract.GROUP_SIZE,
        "minibatch_groups": contract.MINIBATCH_GROUPS,
        "optimizer_updates_per_window": contract.OPTIMIZER_UPDATES_PER_WINDOW,
        "total_windows": contract.TOTAL_WINDOWS,
        "total_optimizer_updates": contract.TOTAL_OPTIMIZER_UPDATES,
    }
    expected_structure = {
        "groups": contract.EXPECTED_RECOMMEND_GROUPS,
        "positive_edges": contract.EXPECTED_POSITIVE_EDGES,
        "unique_group_ids": contract.EXPECTED_RECOMMEND_GROUPS,
        "trie_leaves": contract.EXPECTED_TRIE_LEAVES,
        "trie_a_nodes": contract.EXPECTED_DOMAIN_A_NODES,
        "trie_ab_nodes": contract.EXPECTED_DOMAIN_AB_NODES,
        "effective_groups_per_window": 32,
        "group_size": 16,
        "minibatch_groups": 8,
        "optimizer_updates_per_window": 4,
        "total_windows": 1064,
        "total_optimizer_updates": 4256,
    }
    passed = structure == expected_structure and input_sha == expected_sha
    report = _bound_report(
        "structure",
        signature,
        passed=passed,
        structure=structure,
        structure_sha256=canonical_sha256(structure),
        input_sha256=input_sha,
        expected_input_sha256=expected_sha,
        anchor_fields_present=False,
    )
    _atomic_json(output, report)
    if not passed:
        raise RuntimeError("Structure gate failed.")
    return report


def _gate_groups(groups: Sequence[Any], count: int) -> tuple[Any, ...]:
    if count <= 0 or count > len(groups):
        raise ValueError("Invalid gate group count.")
    return tuple(
        sorted(
            groups,
            key=lambda group: canonical_sha256(
                ["rloo-dapo-gate", 42, group.group_id]
            ),
        )[:count]
    )


def _prompt_requests(bundle: Any, groups: Sequence[Any], config: Mapping[str, Any]):
    return tuple(
        PromptRequest(
            group.group_id,
            tuple(
                encode_prompt(
                    bundle.tokenizer,
                    group.system,
                    group.prompt,
                    cutoff_len=int(config["data"]["cutoff_len"]),
                )
            ),
        )
        for group in groups
    )


def probability_gate(
    config: dict[str, Any],
    signature: dict[str, Any],
    *,
    output: Path,
    device: str | torch.device = "cuda:0",
) -> dict[str, Any]:
    validate_runtime_signature(signature)
    groups, trie = load_groups_and_trie(config)
    selected = _gate_groups(groups, int(config["gates"]["parity_groups"]))
    bundle = load_policy_model(config, device=device)
    grammar = RecommendationGrammar(bundle.tokenizer, trie)
    maximum_delta = 0.0
    maximum_proposal_delta = 0.0
    corrected = 0
    all_legal = True
    all_finite = True
    candidates_run = 0
    scoring_width = grammar_completion_width(grammar)
    for batch_start in range(0, len(selected), contract.PROMPT_BATCH_SIZE):
        raw = selected[batch_start : batch_start + contract.PROMPT_BATCH_SIZE]
        requests = _prompt_requests(bundle, raw, config)
        proposals, _ = rollout_prompt_batch(
            bundle.model,
            requests,
            grammar,
            sampling_nonce=0,
            config=config,
            device=device,
        )
        for proposal in proposals:
            final = canonicalize_prompt_rollout(
                bundle.model,
                proposal,
                grammar,
                sampling_nonce=0,
                temperature=float(config["rollout"]["temperature"]),
                max_difference=float(
                    config["rollout"]["sample_canonical_max_logp_difference"]
                ),
                device=device,
            )
            corrected += final.corrected_candidate_count
            maximum_proposal_delta = max(
                maximum_proposal_delta,
                final.max_proposal_canonical_logp_difference,
            )
            for start in (0, contract.LOSS_CHUNK_SIZE):
                chunk = final.candidates[start : start + contract.LOSS_CHUNK_SIZE]
                with torch.no_grad():
                    scores = score_completions_dense_fixed(
                        bundle.model,
                        final.request.prompt_ids,
                        [candidate.token_ids for candidate in chunk],
                        grammar,
                        device=device,
                        temperature=float(config["rollout"]["temperature"]),
                        batch_rows=contract.LOSS_CHUNK_SIZE,
                        completion_width=scoring_width,
                    )
                for row, candidate in enumerate(chunk):
                    length = len(candidate.token_ids)
                    old = torch.tensor(
                        candidate.old_log_probs,
                        dtype=torch.float32,
                        device=scores.log_probs.device,
                    )
                    mask = torch.tensor(
                        candidate.decision_mask,
                        dtype=torch.bool,
                        device=scores.log_probs.device,
                    )
                    delta = (scores.log_probs[row, :length] - old).abs()[mask]
                    if delta.numel():
                        maximum_delta = max(
                            maximum_delta, float(delta.max().item())
                        )
                    all_finite = all_finite and bool(
                        torch.isfinite(scores.log_probs[row, :length]).all()
                    )
                    try:
                        all_legal = all_legal and (
                            grammar.parse(candidate.token_ids) == candidate.sid
                        )
                    except ValueError:
                        all_legal = False
                    candidates_run += 1
    passed = (
        candidates_run == len(selected) * contract.GROUP_SIZE
        and all_legal
        and all_finite
        and maximum_delta
        <= float(config["rollout"]["canonical_replay_max_logp_difference"])
    )
    report = _bound_report(
        "probability",
        signature,
        passed=passed,
        groups_run=len(selected),
        candidates_run=candidates_run,
        all_legal=all_legal,
        all_finite=all_finite,
        max_sample_canonical_logp_difference=0.0,
        max_canonical_replay_logp_difference=maximum_delta,
        max_proposal_canonical_logp_difference=maximum_proposal_delta,
        corrected_candidates=corrected,
        anchor_groups=0,
        gt_injection_count=0,
        k=contract.NUM_ITERATIONS,
    )
    _atomic_json(output, report)
    if not passed:
        raise RuntimeError("Probability gate failed.")
    return report


def memory_gate(
    config: dict[str, Any],
    signature: dict[str, Any],
    *,
    output: Path,
    device: str | torch.device = "cuda:0",
) -> dict[str, Any]:
    validate_runtime_signature(signature)
    configure_deterministic_runtime(int(config["train"]["seed"]))
    groups, trie = load_groups_and_trie(config)
    bundle = load_policy_model(config, device=device)
    grammar = RecommendationGrammar(bundle.tokenizer, trie)
    longest_group = None
    longest_prompt: tuple[int, ...] = ()
    for group in groups:
        prompt = tuple(
            encode_prompt(
                bundle.tokenizer,
                group.system,
                group.prompt,
                cutoff_len=int(config["data"]["cutoff_len"]),
            )
        )
        if len(prompt) > len(longest_prompt):
            longest_group, longest_prompt = group, prompt
    if longest_group is None:
        raise RuntimeError("No prompt was available for the memory gate.")
    torch_device = torch.device(device)
    torch.cuda.reset_peak_memory_stats(torch_device)
    worst_requests = tuple(
        PromptRequest(f"{longest_group.group_id}:memory:{index}", longest_prompt)
        for index in range(contract.PROMPT_BATCH_SIZE)
    )
    proposals, stats = rollout_prompt_batch(
        bundle.model,
        worst_requests,
        grammar,
        sampling_nonce=0,
        config=config,
        device=torch_device,
    )
    finals = tuple(
        canonicalize_prompt_rollout(
            bundle.model,
            proposal,
            grammar,
            sampling_nonce=0,
            temperature=float(config["rollout"]["temperature"]),
            max_difference=float(
                config["rollout"]["sample_canonical_max_logp_difference"]
            ),
            device=torch_device,
        )
        for proposal in proposals
    )
    chunk = finals[0].candidates[: contract.LOSS_CHUNK_SIZE]
    scoring_width = grammar_completion_width(grammar)
    bundle.model.zero_grad(set_to_none=True)
    scores = score_completions_dense_fixed(
        bundle.model,
        longest_prompt,
        [candidate.token_ids for candidate in chunk],
        grammar,
        device=torch_device,
        temperature=float(config["rollout"]["temperature"]),
        batch_rows=contract.LOSS_CHUNK_SIZE,
        completion_width=scoring_width,
    )
    old = torch.zeros_like(scores.log_probs)
    mask = torch.zeros_like(scores.decision_mask)
    for row, candidate in enumerate(chunk):
        length = len(candidate.old_log_probs)
        old[row, :length] = torch.tensor(candidate.old_log_probs, device=torch_device)
        mask[row, :length] = torch.tensor(candidate.decision_mask, device=torch_device)
    advantages = torch.linspace(-1.0, 1.0, contract.LOSS_CHUNK_SIZE, device=torch_device)
    objective = clipped_rloo_token_sum(scores.log_probs, old, advantages, mask)
    (objective.loss_sum / mask.sum()).backward()
    torch.cuda.synchronize(torch_device)
    peak = torch.cuda.max_memory_reserved(torch_device) / 1024**3
    finite = all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in bundle.model.parameters()
        if parameter.requires_grad
    )
    passed = (
        len(longest_prompt) == int(config["memory"]["max_observed_prompt_length"])
        and peak <= float(config["memory"]["max_reserved_gib"])
        and finite
    )
    report = _bound_report(
        "memory",
        signature,
        passed=passed,
        prompt_group_id=longest_group.group_id,
        prompt_length=len(longest_prompt),
        prompt_batch_size=len(worst_requests),
        peak_reserved_gib=peak,
        max_reserved_gib=float(config["memory"]["max_reserved_gib"]),
        gradients_finite=finite,
        cache_stats=asdict(stats),
    )
    _atomic_json(output, report)
    if not passed:
        raise RuntimeError("Memory gate failed.")
    return report


def throughput_gate(
    config: dict[str, Any],
    signature: dict[str, Any],
    *,
    output: Path,
    device: str | torch.device = "cuda:0",
) -> dict[str, Any]:
    validate_runtime_signature(signature)
    groups, trie = load_groups_and_trie(config)
    selected = _gate_groups(groups, int(config["gates"]["throughput_groups"]))
    bundle = load_policy_model(config, device=device)
    grammar = RecommendationGrammar(bundle.tokenizer, trie)
    requests = _prompt_requests(bundle, selected, config)
    torch.cuda.synchronize(torch.device(device))
    start = time.perf_counter()
    for request in requests:
        rollout_group(
            bundle.model,
            request.prompt_ids,
            grammar,
            group_id=request.group_id,
            epoch_index=0,
            chunk_size=contract.LOSS_CHUNK_SIZE,
            num_candidates=contract.GROUP_SIZE,
            max_completion_length=contract.MAX_COMPLETION_LENGTH,
            device=device,
            temperature=float(config["rollout"]["temperature"]),
        )
    torch.cuda.synchronize(torch.device(device))
    baseline_seconds = time.perf_counter() - start

    proposal_seconds = 0.0
    exact_seconds = 0.0
    for batch_start in range(0, len(requests), contract.PROMPT_BATCH_SIZE):
        batch = requests[batch_start : batch_start + contract.PROMPT_BATCH_SIZE]
        torch.cuda.synchronize(torch.device(device))
        start = time.perf_counter()
        proposals, _ = rollout_prompt_batch(
            bundle.model,
            batch,
            grammar,
            sampling_nonce=0,
            config=config,
            device=device,
        )
        torch.cuda.synchronize(torch.device(device))
        proposal_seconds += time.perf_counter() - start
        start = time.perf_counter()
        for proposal in proposals:
            canonicalize_prompt_rollout(
                bundle.model,
                proposal,
                grammar,
                sampling_nonce=0,
                temperature=float(config["rollout"]["temperature"]),
                max_difference=float(
                    config["rollout"]["sample_canonical_max_logp_difference"]
                ),
                device=device,
            )
        torch.cuda.synchronize(torch.device(device))
        exact_seconds += time.perf_counter() - start
    exact_total = proposal_seconds + exact_seconds
    proposal_speedup = baseline_seconds / proposal_seconds
    exact_speedup = baseline_seconds / exact_total
    passed = exact_speedup >= float(config["gates"]["min_rollout_speedup"])
    report = _bound_report(
        "throughput",
        signature,
        passed=passed,
        groups_run=len(selected),
        rollouts=len(selected) * contract.GROUP_SIZE,
        baseline_seconds=baseline_seconds,
        cache_proposal_seconds=proposal_seconds,
        correction_seconds=exact_seconds,
        exact_cached_rollout_seconds=exact_total,
        cache_proposal_speedup=proposal_speedup,
        exact_cached_rollout_speedup=exact_speedup,
        required_rollout_speedup=float(config["gates"]["min_rollout_speedup"]),
    )
    _atomic_json(output, report)
    if not passed:
        raise RuntimeError("Throughput gate failed.")
    return report


def pilot_gate(
    config: dict[str, Any],
    signature: dict[str, Any],
    *,
    output: Path,
    pilot_dir: Path,
    device: str | torch.device = "cuda:0",
) -> dict[str, Any]:
    validate_runtime_signature(signature)
    initial_hash = sha256_file(Path(config["model"]["sft_adapter"]) / "adapter_model.safetensors")
    comparison_windows = int(config["gates"]["end_to_end_windows"])
    baseline_dir = Path(pilot_dir).with_name(f"{Path(pilot_dir).name}-dense-baseline")
    run_training(
        config,
        signature,
        output_dir=baseline_dir,
        device=device,
        resume=False,
        stop_after_windows=comparison_windows,
        aligned_cache=False,
        dense_scoring=True,
    )
    baseline_rows = [
        json.loads(line)
        for line in (baseline_dir / "windows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    gc.collect()
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    started = time.perf_counter()
    summary = run_training(
        config,
        signature,
        output_dir=pilot_dir,
        device=device,
        resume=False,
        stop_after_windows=int(config["gates"]["pilot_windows"]),
    )
    elapsed = time.perf_counter() - started
    checkpoint = Path(summary["last_checkpoint"])
    final_hash = sha256_file(checkpoint / "adapter_model.safetensors")
    window_rows = [
        json.loads(line)
        for line in (Path(pilot_dir) / "windows.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    groups = [group_id for row in window_rows for group_id in row["group_ids"]]
    peak = max(float(row["peak_reserved_gib"]) for row in window_rows)
    updates = sum(int(row["optimizer_updates"]) for row in window_rows)
    baseline_window_seconds = sum(
        float(row["window_seconds"]) for row in baseline_rows
    )
    optimized_window_seconds = sum(
        float(row["window_seconds"]) for row in window_rows[:comparison_windows]
    )
    end_to_end_speedup = baseline_window_seconds / optimized_window_seconds
    passed = (
        len(baseline_rows) == comparison_windows
        and len(window_rows) == int(config["gates"]["pilot_windows"])
        and len(groups) == len(window_rows) * contract.EFFECTIVE_GROUPS_PER_WINDOW
        and updates == len(window_rows) * contract.OPTIMIZER_UPDATES_PER_WINDOW
        and initial_hash != final_hash
        and peak <= float(config["memory"]["max_reserved_gib"])
        and all(row["anchor_groups"] == 0 for row in window_rows)
        and all(row["gt_injection_count"] == 0 for row in window_rows)
        and all(
            int(row["effective_groups"]) == contract.EFFECTIVE_GROUPS_PER_WINDOW
            and int(row["optimizer_updates"])
            == contract.OPTIMIZER_UPDATES_PER_WINDOW
            for row in baseline_rows
        )
        and end_to_end_speedup >= float(config["gates"]["min_window_speedup"])
    )
    report = _bound_report(
        "pilot",
        signature,
        passed=passed,
        windows=len(window_rows),
        effective_groups=len(groups),
        optimizer_updates=updates,
        unique_group_occurrences=len(groups),
        initial_adapter_sha256=initial_hash,
        final_adapter_sha256=final_hash,
        parameter_changed=initial_hash != final_hash,
        peak_reserved_gib=peak,
        elapsed_seconds=elapsed,
        end_to_end_windows=comparison_windows,
        dense_baseline_dir=str(baseline_dir),
        baseline_window_seconds=baseline_window_seconds,
        optimized_window_seconds=optimized_window_seconds,
        end_to_end_window_speedup=end_to_end_speedup,
        required_window_speedup=float(config["gates"]["min_window_speedup"]),
        run_summary=summary,
        anchor_groups=0,
        gt_injection_count=0,
        k=contract.NUM_ITERATIONS,
    )
    _atomic_json(output, report)
    if not passed:
        raise RuntimeError("Two-window pilot gate failed.")
    return report


def load_and_validate_gate_reports(
    config: Mapping[str, Any],
    signature: Mapping[str, Any],
    paths: Mapping[str, Path],
) -> dict[str, dict[str, Any]]:
    required = {"structure", "probability", "memory", "throughput", "pilot"}
    if set(paths) != required:
        raise ValueError("Formal training requires all five gate reports.")
    reports: dict[str, dict[str, Any]] = {}
    for name in sorted(required):
        value = json.loads(Path(paths[name]).read_text(encoding="utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != GATE_SCHEMA_VERSION
            or value.get("gate") != name
            or value.get("passed") is not True
            or value.get("runtime_signature_sha256") != signature["sha256"]
        ):
            raise RuntimeError(f"{name} gate report is invalid or belongs to another run.")
        reports[name] = value
    if reports["structure"].get("input_sha256") != expected_input_sha256():
        raise RuntimeError("Structure gate input hashes differ from the frozen contract.")
    if float(reports["probability"]["max_canonical_replay_logp_difference"]) > float(
        config["rollout"]["canonical_replay_max_logp_difference"]
    ):
        raise RuntimeError("Probability report exceeds the replay threshold.")
    if float(reports["memory"]["peak_reserved_gib"]) > float(
        config["memory"]["max_reserved_gib"]
    ):
        raise RuntimeError("Memory report exceeds the reserved-memory threshold.")
    if float(reports["throughput"]["exact_cached_rollout_speedup"]) < float(
        config["gates"]["min_rollout_speedup"]
    ):
        raise RuntimeError("Throughput report is below the rollout threshold.")
    if float(reports["pilot"].get("end_to_end_window_speedup", 0.0)) < float(
        config["gates"]["min_window_speedup"]
    ):
        raise RuntimeError("Pilot report is below the end-to-end window threshold.")
    if reports["pilot"].get("parameter_changed") is not True:
        raise RuntimeError("Pilot report did not prove a parameter update.")
    return reports


__all__ = [
    "collect_input_sha256",
    "expected_input_sha256",
    "load_and_validate_gate_reports",
    "memory_gate",
    "pilot_gate",
    "probability_gate",
    "structure_gate",
    "throughput_gate",
]
