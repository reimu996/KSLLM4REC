"""Measured preflight gates bound to the exact RLOO-DAPO runtime signature."""

from __future__ import annotations

from dataclasses import asdict
import gc
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

import torch

from ksllm4rec_grpo.constraint import RecommendationGrammar
from ksllm4rec_grpo.prompt import encode_prompt
from ksllm4rec_rloo.integrity import (
    canonical_sha256,
    sha256_file,
    snapshot_directory,
    verify_directory_snapshot,
)
from ksllm4rec_rloo.modeling import load_policy_model
from ksllm4rec_rloo.rollout import rollout_group

from . import contract
from .device import gpu_identity, validate_gpu_identity
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
_VOLATILE_WINDOW_FIELDS = {
    "collection_seconds",
    "training_seconds",
    "window_seconds",
    "peak_reserved_gib",
    "collection_peak_reserved_gib",
    "training_peak_reserved_gib",
}


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


def _deterministic_window_rows(path: Path) -> list[dict[str, Any]]:
    rows = [
        json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()
    ]
    return [
        {key: value for key, value in row.items() if key not in _VOLATILE_WINDOW_FIELDS}
        for row in rows
    ]


def _adapter_config_semantic_sha256(path: Path) -> str:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or not isinstance(value.get("target_modules"), list):
        raise RuntimeError("Pilot adapter config is invalid.")
    normalized = dict(value)
    normalized["target_modules"] = sorted(str(item) for item in value["target_modules"])
    return canonical_sha256(normalized)


def expected_input_sha256(
    config: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    profile_name = (
        contract.PROFILE if config is None else str(config.get("profile", ""))
    )
    profile = contract.frozen_profile(profile_name)
    return {
        "base_model": contract.BASE_MODEL_SHA256,
        "sft_adapter": profile.sft_adapter_sha256,
        "sft_adapter_config": profile.sft_config_sha256,
        "tokenizer": profile.tokenizer_sha256,
        "tokenizer_config": profile.tokenizer_config_sha256,
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
        "tokenizer_config": sha256_file(tokenizer / "tokenizer_config.json"),
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
    expected_sha = expected_input_sha256(config)
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
            key=lambda group: canonical_sha256(["rloo-dapo-gate", 42, group.group_id]),
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
    measured_gpu = gpu_identity(device)
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
                        maximum_delta = max(maximum_delta, float(delta.max().item()))
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
        gpu_identity=measured_gpu,
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
    measured_gpu = gpu_identity(device)
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
    advantages = torch.linspace(
        -1.0, 1.0, contract.LOSS_CHUNK_SIZE, device=torch_device
    )
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
        gpu_identity=measured_gpu,
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
    measured_gpu = gpu_identity(device)
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
        gpu_identity=measured_gpu,
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
    measured_gpu = gpu_identity(device)
    initial_hash = sha256_file(
        Path(config["model"]["sft_adapter"]) / "adapter_model.safetensors"
    )
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
        for line in (baseline_dir / "windows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
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
    recovery_dir = Path(pilot_dir).with_name(f"{Path(pilot_dir).name}-resume-check")
    gc.collect()
    if torch.cuda.is_available() and torch.device(device).type == "cuda":
        torch.cuda.empty_cache()
    recovery_started = time.perf_counter()
    worker_command = [
        sys.executable,
        "-m",
        "ksllm4rec_rloo_dapo.recovery_worker",
        "--config",
        str(signature["inputs"]["config"]["source_file"]["path"]),
        "--expected-signature",
        str(signature["sha256"]),
        "--output-dir",
        str(recovery_dir),
        "--device",
        str(device),
    ]
    worker_environment = os.environ.copy()
    worker_environment["PYTHONHASHSEED"] = "random"
    subprocess.run(worker_command, check=True, env=worker_environment)
    subprocess.run([*worker_command, "--resume"], check=True, env=worker_environment)
    recovery_summary = json.loads(
        (recovery_dir / "run_summary.json").read_text(encoding="utf-8")
    )
    recovery_elapsed = time.perf_counter() - recovery_started
    checkpoint = Path(pilot_dir) / summary["last_checkpoint"]
    recovery_checkpoint = recovery_dir / recovery_summary["last_checkpoint"]
    final_hash = sha256_file(checkpoint / "adapter_model.safetensors")
    recovery_final_hash = sha256_file(recovery_checkpoint / "adapter_model.safetensors")
    continuous_groups_hash = sha256_file(Path(pilot_dir) / "groups.jsonl")
    recovery_groups_hash = sha256_file(recovery_dir / "groups.jsonl")
    continuous_state_hash = sha256_file(checkpoint / "training_state.pt")
    recovery_state_hash = sha256_file(recovery_checkpoint / "training_state.pt")
    continuous_config_hash = _adapter_config_semantic_sha256(
        checkpoint / "adapter_config.json"
    )
    recovery_config_hash = _adapter_config_semantic_sha256(
        recovery_checkpoint / "adapter_config.json"
    )
    recovery_exact = (
        _deterministic_window_rows(Path(pilot_dir) / "windows.jsonl")
        == _deterministic_window_rows(recovery_dir / "windows.jsonl")
        and continuous_groups_hash == recovery_groups_hash
        and final_hash == recovery_final_hash
        and continuous_state_hash == recovery_state_hash
        and continuous_config_hash == recovery_config_hash
    )
    window_rows = [
        json.loads(line)
        for line in (Path(pilot_dir) / "windows.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    groups = [group_id for row in window_rows for group_id in row["group_ids"]]
    peak = max(float(row["peak_reserved_gib"]) for row in window_rows)
    updates = sum(int(row["optimizer_updates"]) for row in window_rows)
    baseline_window_seconds = sum(float(row["window_seconds"]) for row in baseline_rows)
    optimized_window_seconds = sum(
        float(row["window_seconds"]) for row in window_rows[:comparison_windows]
    )
    end_to_end_speedup = baseline_window_seconds / optimized_window_seconds
    pilot_files = snapshot_directory(Path(pilot_dir))
    baseline_files = snapshot_directory(baseline_dir)
    recovery_files = snapshot_directory(recovery_dir)
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
            and int(row["optimizer_updates"]) == contract.OPTIMIZER_UPDATES_PER_WINDOW
            for row in baseline_rows
        )
        and end_to_end_speedup >= float(config["gates"]["min_window_speedup"])
        and optimized_window_seconds <= contract.MAX_OPTIMIZED_WINDOW_SECONDS
        and recovery_exact
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
        pilot_dir=str(Path(pilot_dir).resolve()),
        dense_baseline_dir=str(baseline_dir),
        recovery_check_dir=str(recovery_dir.resolve()),
        pilot_files=pilot_files,
        dense_baseline_files=baseline_files,
        recovery_check_files=recovery_files,
        baseline_window_seconds=baseline_window_seconds,
        optimized_window_seconds=optimized_window_seconds,
        max_optimized_window_seconds=contract.MAX_OPTIMIZED_WINDOW_SECONDS,
        end_to_end_window_speedup=end_to_end_speedup,
        required_window_speedup=float(config["gates"]["min_window_speedup"]),
        run_summary=summary,
        recovery_run_summary=recovery_summary,
        recovery_elapsed_seconds=recovery_elapsed,
        recovery_exact=recovery_exact,
        recovery_processes=2,
        continuous_groups_sha256=continuous_groups_hash,
        recovery_groups_sha256=recovery_groups_hash,
        continuous_training_state_sha256=continuous_state_hash,
        recovery_training_state_sha256=recovery_state_hash,
        continuous_adapter_config_semantic_sha256=continuous_config_hash,
        recovery_adapter_config_semantic_sha256=recovery_config_hash,
        recovery_final_adapter_sha256=recovery_final_hash,
        anchor_groups=0,
        gt_injection_count=0,
        k=contract.NUM_ITERATIONS,
        gpu_identity=measured_gpu,
    )
    _atomic_json(output, report)
    if not passed:
        raise RuntimeError("Two-window pilot gate failed.")
    return report


def _require_fields(gate: str, report: Mapping[str, Any], required: set[str]) -> None:
    missing = sorted(required - set(report))
    if missing:
        raise RuntimeError(f"{gate} gate report is missing fields: {missing}.")


def _require_exact(
    gate: str, report: Mapping[str, Any], key: str, expected: Any
) -> None:
    actual = report[key]
    if type(actual) is not type(expected) or actual != expected:
        raise RuntimeError(
            f"{gate} gate field {key!r} is inconsistent with the contract."
        )


def _finite_number(
    gate: str,
    report: Mapping[str, Any],
    key: str,
    *,
    positive: bool = False,
) -> float:
    value = report[key]
    if type(value) not in (int, float) or not math.isfinite(value):
        raise RuntimeError(f"{gate} gate field {key!r} must be finite.")
    number = float(value)
    if (positive and number <= 0.0) or (not positive and number < 0.0):
        raise RuntimeError(f"{gate} gate field {key!r} has an invalid sign.")
    return number


def _nonnegative_int(gate: str, report: Mapping[str, Any], key: str) -> int:
    value = report[key]
    if type(value) is not int or value < 0:
        raise RuntimeError(f"{gate} gate field {key!r} must be a non-negative integer.")
    return value


def _require_derived(gate: str, key: str, actual: float, expected: float) -> None:
    if not math.isclose(actual, expected, rel_tol=1.0e-12, abs_tol=1.0e-12):
        raise RuntimeError(f"{gate} gate derived field {key!r} was not reproduced.")


def _reject_non_finite(value: Any, *, location: str) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise RuntimeError(f"Gate report contains NaN or Inf at {location}.")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _reject_non_finite(child, location=f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_non_finite(child, location=f"{location}[{index}]")


def _load_report(path: Path, gate: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise RuntimeError(f"{gate} gate report is not valid JSON.") from exc
    if not isinstance(value, dict):
        raise RuntimeError(f"{gate} gate report must be a JSON object.")
    _reject_non_finite(value, location=gate)
    return value


def _validate_structure_report(
    config: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    gate = "structure"
    _require_fields(
        gate,
        report,
        {
            "structure",
            "structure_sha256",
            "input_sha256",
            "expected_input_sha256",
            "anchor_fields_present",
        },
    )
    expected_sha = expected_input_sha256(config)
    if report["input_sha256"] != expected_sha:
        raise RuntimeError(
            "Structure gate input hashes differ from the frozen contract."
        )
    if report["expected_input_sha256"] != expected_sha:
        raise RuntimeError("Structure gate expected input hashes were altered.")
    expected_structure = {
        "groups": int(config["data"]["groups"]),
        "positive_edges": int(config["data"]["positives"]),
        "unique_group_ids": int(config["data"]["groups"]),
        "trie_leaves": int(config["trie"]["unique_sids"]),
        "trie_a_nodes": int(config["trie"]["domain_a_nodes"]),
        "trie_ab_nodes": int(config["trie"]["domain_ab_nodes"]),
        "effective_groups_per_window": int(
            config["sampling"]["effective_groups_per_window"]
        ),
        "group_size": int(config["rollout"]["num_generations"]),
        "minibatch_groups": int(config["loss"]["minibatch_groups"]),
        "optimizer_updates_per_window": int(
            config["train"]["optimizer_updates_per_window"]
        ),
        "total_windows": int(config["train"]["total_windows"]),
        "total_optimizer_updates": int(config["train"]["total_optimizer_updates"]),
    }
    structure = report["structure"]
    if not isinstance(structure, dict) or set(structure) != set(expected_structure):
        raise RuntimeError("Structure gate counts have an invalid schema.")
    if any(
        type(structure[key]) is not int or structure[key] != expected
        for key, expected in expected_structure.items()
    ):
        raise RuntimeError("Structure gate counts differ from the approved contract.")
    if report["structure_sha256"] != canonical_sha256(structure):
        raise RuntimeError("Structure gate count digest is inconsistent.")
    _require_exact(gate, report, "anchor_fields_present", False)


def _validate_probability_report(
    config: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    gate = "probability"
    _require_fields(
        gate,
        report,
        {
            "groups_run",
            "candidates_run",
            "all_legal",
            "all_finite",
            "max_sample_canonical_logp_difference",
            "max_canonical_replay_logp_difference",
            "max_proposal_canonical_logp_difference",
            "corrected_candidates",
            "anchor_groups",
            "gt_injection_count",
            "k",
            "gpu_identity",
        },
    )
    groups = int(config["gates"]["parity_groups"])
    candidates = groups * int(config["rollout"]["num_generations"])
    _require_exact(gate, report, "groups_run", groups)
    _require_exact(gate, report, "candidates_run", candidates)
    _require_exact(gate, report, "all_legal", True)
    _require_exact(gate, report, "all_finite", True)
    sample_delta = _finite_number(gate, report, "max_sample_canonical_logp_difference")
    replay_delta = _finite_number(gate, report, "max_canonical_replay_logp_difference")
    proposal_delta = _finite_number(
        gate, report, "max_proposal_canonical_logp_difference"
    )
    sample_limit = float(config["rollout"]["sample_canonical_max_logp_difference"])
    replay_limit = float(config["rollout"]["canonical_replay_max_logp_difference"])
    if sample_delta > sample_limit:
        raise RuntimeError("Probability report exceeds the sample threshold.")
    if replay_delta > replay_limit:
        raise RuntimeError("Probability report exceeds the replay threshold.")
    corrected = _nonnegative_int(gate, report, "corrected_candidates")
    if corrected > candidates:
        raise RuntimeError("Probability report corrected more candidates than it ran.")
    if corrected == 0 and proposal_delta > sample_limit:
        raise RuntimeError("Probability report omitted required corrections.")
    _require_exact(gate, report, "anchor_groups", 0)
    _require_exact(gate, report, "gt_injection_count", 0)
    _require_exact(gate, report, "k", contract.NUM_ITERATIONS)


def _validate_memory_report(
    config: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    gate = "memory"
    _require_fields(
        gate,
        report,
        {
            "prompt_group_id",
            "prompt_length",
            "prompt_batch_size",
            "peak_reserved_gib",
            "max_reserved_gib",
            "gradients_finite",
            "cache_stats",
            "gpu_identity",
        },
    )
    if not isinstance(report["prompt_group_id"], str) or not report["prompt_group_id"]:
        raise RuntimeError("Memory gate prompt_group_id must be non-empty.")
    prompt_batch = int(config["rollout"]["prompt_batch_size"])
    _require_exact(
        gate,
        report,
        "prompt_length",
        int(config["memory"]["max_observed_prompt_length"]),
    )
    _require_exact(gate, report, "prompt_batch_size", prompt_batch)
    peak = _finite_number(gate, report, "peak_reserved_gib")
    maximum = float(config["memory"]["max_reserved_gib"])
    _require_exact(gate, report, "max_reserved_gib", maximum)
    if peak > maximum:
        raise RuntimeError("Memory report exceeds the reserved-memory threshold.")
    _require_exact(gate, report, "gradients_finite", True)
    cache = report["cache_stats"]
    cache_fields = {
        "prompt_count",
        "rollout_count",
        "prefill_calls",
        "decode_calls",
        "cache_forks",
        "requested_active_sequences",
        "resolved_active_sequences",
        "estimated_peak_cache_bytes",
        "fallback_prompt_count",
    }
    if not isinstance(cache, dict) or not cache_fields.issubset(cache):
        raise RuntimeError("Memory gate cache_stats has an invalid schema.")
    expected_cache_counts = {
        "prompt_count": prompt_batch,
        "rollout_count": prompt_batch * int(config["rollout"]["num_generations"]),
        "prefill_calls": prompt_batch,
        "cache_forks": 0,
        "requested_active_sequences": int(config["rollout"]["max_active_sequences"]),
        "resolved_active_sequences": int(config["rollout"]["max_active_sequences"]),
        "fallback_prompt_count": 0,
    }
    if any(
        type(cache[key]) is not int or cache[key] != expected
        for key, expected in expected_cache_counts.items()
    ):
        raise RuntimeError("Memory gate cache counts differ from the contract.")
    decode_calls = cache["decode_calls"]
    if (
        type(decode_calls) is not int
        or decode_calls <= 0
        or decode_calls > prompt_batch * int(config["rollout"]["max_completion_length"])
    ):
        raise RuntimeError("Memory gate decode call count is invalid.")
    cache_bytes = cache["estimated_peak_cache_bytes"]
    cache_budget = float(config["rollout"]["kv_cache_budget_gib"]) * 1024**3
    if type(cache_bytes) is not int or not 0 < cache_bytes <= cache_budget:
        raise RuntimeError("Memory gate cache byte count is invalid.")


def _validate_throughput_report(
    config: Mapping[str, Any], report: Mapping[str, Any]
) -> None:
    gate = "throughput"
    _require_fields(
        gate,
        report,
        {
            "groups_run",
            "rollouts",
            "baseline_seconds",
            "cache_proposal_seconds",
            "correction_seconds",
            "exact_cached_rollout_seconds",
            "cache_proposal_speedup",
            "exact_cached_rollout_speedup",
            "required_rollout_speedup",
            "gpu_identity",
        },
    )
    groups = int(config["gates"]["throughput_groups"])
    _require_exact(gate, report, "groups_run", groups)
    _require_exact(
        gate,
        report,
        "rollouts",
        groups * int(config["rollout"]["num_generations"]),
    )
    baseline = _finite_number(gate, report, "baseline_seconds", positive=True)
    proposal = _finite_number(gate, report, "cache_proposal_seconds", positive=True)
    correction = _finite_number(gate, report, "correction_seconds", positive=True)
    exact = _finite_number(gate, report, "exact_cached_rollout_seconds", positive=True)
    proposal_speedup = _finite_number(
        gate, report, "cache_proposal_speedup", positive=True
    )
    exact_speedup = _finite_number(
        gate, report, "exact_cached_rollout_speedup", positive=True
    )
    _require_derived(gate, "exact_cached_rollout_seconds", exact, proposal + correction)
    _require_derived(
        gate, "cache_proposal_speedup", proposal_speedup, baseline / proposal
    )
    _require_derived(
        gate, "exact_cached_rollout_speedup", exact_speedup, baseline / exact
    )
    threshold = float(config["gates"]["min_rollout_speedup"])
    _require_exact(gate, report, "required_rollout_speedup", threshold)
    if exact_speedup < threshold:
        raise RuntimeError("Throughput report is below the rollout threshold.")


def _validate_pilot_report(
    config: Mapping[str, Any],
    signature: Mapping[str, Any],
    report: Mapping[str, Any],
) -> None:
    gate = "pilot"
    _require_fields(
        gate,
        report,
        {
            "windows",
            "effective_groups",
            "optimizer_updates",
            "unique_group_occurrences",
            "initial_adapter_sha256",
            "final_adapter_sha256",
            "parameter_changed",
            "peak_reserved_gib",
            "elapsed_seconds",
            "end_to_end_windows",
            "pilot_dir",
            "dense_baseline_dir",
            "recovery_check_dir",
            "pilot_files",
            "dense_baseline_files",
            "recovery_check_files",
            "baseline_window_seconds",
            "optimized_window_seconds",
            "max_optimized_window_seconds",
            "end_to_end_window_speedup",
            "required_window_speedup",
            "run_summary",
            "recovery_run_summary",
            "recovery_elapsed_seconds",
            "recovery_exact",
            "recovery_processes",
            "continuous_groups_sha256",
            "recovery_groups_sha256",
            "continuous_training_state_sha256",
            "recovery_training_state_sha256",
            "continuous_adapter_config_semantic_sha256",
            "recovery_adapter_config_semantic_sha256",
            "recovery_final_adapter_sha256",
            "anchor_groups",
            "gt_injection_count",
            "k",
            "gpu_identity",
        },
    )
    windows = int(config["gates"]["pilot_windows"])
    effective_groups = windows * int(config["sampling"]["effective_groups_per_window"])
    updates = windows * int(config["train"]["optimizer_updates_per_window"])
    _require_exact(gate, report, "windows", windows)
    _require_exact(gate, report, "effective_groups", effective_groups)
    _require_exact(gate, report, "unique_group_occurrences", effective_groups)
    _require_exact(gate, report, "optimizer_updates", updates)
    expected_adapter = expected_input_sha256(config)["sft_adapter"]
    if report["initial_adapter_sha256"] != expected_adapter:
        raise RuntimeError("Pilot report started from the wrong SFT adapter.")
    final_adapter = report["final_adapter_sha256"]
    if (
        not isinstance(final_adapter, str)
        or re.fullmatch(r"[0-9a-f]{64}", final_adapter) is None
        or final_adapter == expected_adapter
    ):
        raise RuntimeError("Pilot report did not prove a distinct final adapter.")
    _require_exact(gate, report, "parameter_changed", True)
    peak = _finite_number(gate, report, "peak_reserved_gib")
    if peak > float(config["memory"]["max_reserved_gib"]):
        raise RuntimeError("Pilot report exceeds the reserved-memory threshold.")
    elapsed = _finite_number(gate, report, "elapsed_seconds", positive=True)
    comparison_windows = int(config["gates"]["end_to_end_windows"])
    _require_exact(gate, report, "end_to_end_windows", comparison_windows)
    if comparison_windows <= 0 or comparison_windows > windows:
        raise RuntimeError("Pilot comparison window count is invalid.")
    for key in ("pilot_dir", "dense_baseline_dir", "recovery_check_dir"):
        value = report[key]
        if not isinstance(value, str) or not value or not Path(value).is_absolute():
            raise RuntimeError(f"Pilot artifact path {key!r} is invalid.")
    baseline = _finite_number(gate, report, "baseline_window_seconds", positive=True)
    optimized = _finite_number(gate, report, "optimized_window_seconds", positive=True)
    _require_exact(
        gate,
        report,
        "max_optimized_window_seconds",
        contract.MAX_OPTIMIZED_WINDOW_SECONDS,
    )
    if optimized > contract.MAX_OPTIMIZED_WINDOW_SECONDS:
        raise RuntimeError("Pilot optimized window exceeds the 10-minute limit.")
    speedup = _finite_number(gate, report, "end_to_end_window_speedup", positive=True)
    if elapsed < optimized:
        raise RuntimeError("Pilot elapsed time is shorter than its measured windows.")
    _require_derived(gate, "end_to_end_window_speedup", speedup, baseline / optimized)
    threshold = float(config["gates"]["min_window_speedup"])
    _require_exact(gate, report, "required_window_speedup", threshold)
    if speedup < threshold:
        raise RuntimeError("Pilot report is below the end-to-end window threshold.")
    _require_exact(gate, report, "anchor_groups", 0)
    _require_exact(gate, report, "gt_injection_count", 0)
    _require_exact(gate, report, "k", contract.NUM_ITERATIONS)

    summary = report["run_summary"]
    summary_fields = {
        "schema_version",
        "completed_windows",
        "optimizer_update_step",
        "windows_run_this_invocation",
        "source_cursor",
        "last_checkpoint",
        "complete",
        "anchor_groups",
        "gt_injection_count",
        "k",
    }
    if not isinstance(summary, dict) or not summary_fields.issubset(summary):
        raise RuntimeError("Pilot run_summary has an invalid schema.")
    summary_expected = {
        "schema_version": 1,
        "completed_windows": windows,
        "optimizer_update_step": updates,
        "windows_run_this_invocation": windows,
        "complete": windows == int(config["train"]["total_windows"]),
        "anchor_groups": 0,
        "gt_injection_count": 0,
        "k": contract.NUM_ITERATIONS,
    }
    if any(
        type(summary[key]) is not type(expected) or summary[key] != expected
        for key, expected in summary_expected.items()
    ):
        raise RuntimeError("Pilot run_summary disagrees with the pilot report.")
    cursor = summary["source_cursor"]
    if (
        not isinstance(cursor, dict)
        or set(cursor) != {"cycle_index", "offset"}
        or any(type(value) is not int or value < 0 for value in cursor.values())
    ):
        raise RuntimeError("Pilot run_summary source cursor is invalid.")
    checkpoint = summary["last_checkpoint"]
    expected_checkpoint = f"checkpoint-window-{windows:06d}-update-{updates:06d}"
    checkpoint_relative = Path(checkpoint) if isinstance(checkpoint, str) else None
    if (
        checkpoint_relative is None
        or checkpoint_relative.is_absolute()
        or ".." in checkpoint_relative.parts
        or checkpoint_relative.name != expected_checkpoint
    ):
        raise RuntimeError("Pilot run_summary checkpoint disagrees with its counts.")

    pilot_root = Path(report["pilot_dir"]).resolve()
    baseline_root = Path(report["dense_baseline_dir"]).resolve()
    recovery_root = Path(report["recovery_check_dir"]).resolve()
    try:
        verify_directory_snapshot(pilot_root, report["pilot_files"])
        verify_directory_snapshot(baseline_root, report["dense_baseline_files"])
        verify_directory_snapshot(recovery_root, report["recovery_check_files"])
    except (OSError, TypeError, ValueError, RuntimeError) as exc:
        raise RuntimeError("Pilot artifact snapshot is missing or changed.") from exc

    def artifact_json(root: Path, name: str) -> dict[str, Any]:
        try:
            value = json.loads((root / name).read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Pilot artifact {name} is unreadable.") from exc
        if not isinstance(value, dict):
            raise RuntimeError(f"Pilot artifact {name} must be an object.")
        _reject_non_finite(value, location=f"pilot.{name}")
        return value

    def artifact_jsonl(root: Path, name: str) -> list[dict[str, Any]]:
        try:
            lines = (root / name).read_text(encoding="utf-8").splitlines()
            values = [json.loads(line) for line in lines]
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"Pilot artifact {name} is unreadable.") from exc
        if any(not isinstance(value, dict) for value in values):
            raise RuntimeError(f"Pilot artifact {name} has a non-object row.")
        _reject_non_finite(values, location=f"pilot.{name}")
        return values

    for root in (pilot_root, baseline_root, recovery_root):
        if artifact_json(root, "runtime_signature.json") != signature:
            raise RuntimeError("Pilot artifact runtime signature differs.")
        if artifact_json(root, "resolved_config.json") != config:
            raise RuntimeError("Pilot artifact resolved config differs.")
    if artifact_json(pilot_root, "run_summary.json") != summary:
        raise RuntimeError("Pilot artifact run summary differs from its report.")
    recovery_summary = report["recovery_run_summary"]
    if (
        not isinstance(recovery_summary, dict)
        or artifact_json(recovery_root, "run_summary.json") != recovery_summary
        or recovery_summary.get("completed_windows") != windows
        or recovery_summary.get("optimizer_update_step") != updates
        or recovery_summary.get("windows_run_this_invocation") != 1
        or recovery_summary.get("anchor_groups") != 0
        or recovery_summary.get("gt_injection_count") != 0
        or recovery_summary.get("k") != contract.NUM_ITERATIONS
    ):
        raise RuntimeError("Recovery pilot artifact summary is invalid.")
    _finite_number(gate, report, "recovery_elapsed_seconds", positive=True)
    _require_exact(gate, report, "recovery_exact", True)
    _require_exact(gate, report, "recovery_processes", 2)
    baseline_summary = artifact_json(baseline_root, "run_summary.json")
    baseline_updates = comparison_windows * int(
        config["train"]["optimizer_updates_per_window"]
    )
    if (
        baseline_summary.get("completed_windows") != comparison_windows
        or baseline_summary.get("optimizer_update_step") != baseline_updates
        or baseline_summary.get("windows_run_this_invocation") != comparison_windows
        or baseline_summary.get("anchor_groups") != 0
        or baseline_summary.get("gt_injection_count") != 0
        or baseline_summary.get("k") != contract.NUM_ITERATIONS
    ):
        raise RuntimeError("Dense pilot artifact summary is invalid.")

    pilot_windows = artifact_jsonl(pilot_root, "windows.jsonl")
    baseline_windows = artifact_jsonl(baseline_root, "windows.jsonl")
    pilot_groups = artifact_jsonl(pilot_root, "groups.jsonl")
    baseline_groups = artifact_jsonl(baseline_root, "groups.jsonl")
    if len(pilot_windows) != windows or len(baseline_windows) != comparison_windows:
        raise RuntimeError("Pilot artifact window logs have the wrong row count.")
    if len(pilot_groups) != effective_groups or len(baseline_groups) != (
        comparison_windows * int(config["sampling"]["effective_groups_per_window"])
    ):
        raise RuntimeError("Pilot artifact group logs have the wrong row count.")
    for expected_index, row in enumerate(pilot_windows):
        group_ids = row.get("group_ids")
        if (
            row.get("window_index") != expected_index
            or row.get("effective_groups")
            != int(config["sampling"]["effective_groups_per_window"])
            or row.get("optimizer_updates")
            != int(config["train"]["optimizer_updates_per_window"])
            or not isinstance(group_ids, list)
            or len(group_ids) != int(config["sampling"]["effective_groups_per_window"])
            or len(set(group_ids)) != len(group_ids)
            or row.get("anchor_groups") != 0
            or row.get("gt_injection_count") != 0
            or row.get("k") != contract.NUM_ITERATIONS
        ):
            raise RuntimeError("Optimized pilot artifact window row is invalid.")
    if any(
        row.get("effective") is not True
        or row.get("anchor_groups") != 0
        or row.get("gt_injection_count") != 0
        or row.get("k") != contract.NUM_ITERATIONS
        for row in pilot_groups
    ):
        raise RuntimeError("Optimized pilot artifact group row is invalid.")

    checkpoint_root = (pilot_root / checkpoint_relative).resolve()
    try:
        checkpoint_root.relative_to(pilot_root)
    except ValueError as exc:
        raise RuntimeError("Pilot checkpoint escapes the pilot directory.") from exc
    checkpoint_weights = checkpoint_root / "adapter_model.safetensors"
    if (
        not checkpoint_weights.is_file()
        or sha256_file(checkpoint_weights) != report["final_adapter_sha256"]
    ):
        raise RuntimeError("Pilot checkpoint adapter differs from its report.")

    recovery_checkpoint_value = recovery_summary.get("last_checkpoint")
    recovery_checkpoint_relative = (
        Path(recovery_checkpoint_value)
        if isinstance(recovery_checkpoint_value, str)
        else None
    )
    if (
        recovery_checkpoint_relative is None
        or recovery_checkpoint_relative.is_absolute()
        or ".." in recovery_checkpoint_relative.parts
        or recovery_checkpoint_relative.name != expected_checkpoint
    ):
        raise RuntimeError("Recovery pilot checkpoint path is invalid.")
    recovery_checkpoint_root = (recovery_root / recovery_checkpoint_relative).resolve()
    try:
        recovery_checkpoint_root.relative_to(recovery_root)
    except ValueError as exc:
        raise RuntimeError("Recovery pilot checkpoint escapes its directory.") from exc
    measured_hashes = {
        "continuous_groups_sha256": sha256_file(pilot_root / "groups.jsonl"),
        "recovery_groups_sha256": sha256_file(recovery_root / "groups.jsonl"),
        "continuous_training_state_sha256": sha256_file(
            checkpoint_root / "training_state.pt"
        ),
        "recovery_training_state_sha256": sha256_file(
            recovery_checkpoint_root / "training_state.pt"
        ),
        "recovery_final_adapter_sha256": sha256_file(
            recovery_checkpoint_root / "adapter_model.safetensors"
        ),
        "continuous_adapter_config_semantic_sha256": (
            _adapter_config_semantic_sha256(checkpoint_root / "adapter_config.json")
        ),
        "recovery_adapter_config_semantic_sha256": (
            _adapter_config_semantic_sha256(
                recovery_checkpoint_root / "adapter_config.json"
            )
        ),
    }
    if any(report[key] != value for key, value in measured_hashes.items()):
        raise RuntimeError("Recovery pilot content hashes differ from its report.")
    if not (
        measured_hashes["continuous_groups_sha256"]
        == measured_hashes["recovery_groups_sha256"]
        and measured_hashes["continuous_training_state_sha256"]
        == measured_hashes["recovery_training_state_sha256"]
        and measured_hashes["continuous_adapter_config_semantic_sha256"]
        == measured_hashes["recovery_adapter_config_semantic_sha256"]
        and report["final_adapter_sha256"]
        == measured_hashes["recovery_final_adapter_sha256"]
        and _deterministic_window_rows(pilot_root / "windows.jsonl")
        == _deterministic_window_rows(recovery_root / "windows.jsonl")
    ):
        raise RuntimeError("Interrupted and continuous pilots are not byte-exact.")


def load_and_validate_gate_reports(
    config: Mapping[str, Any],
    signature: Mapping[str, Any],
    paths: Mapping[str, Path],
    *,
    expected_gpu_identity: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    validate_runtime_signature(dict(signature))
    required = {"structure", "probability", "memory", "throughput", "pilot"}
    if set(paths) != required:
        raise ValueError("Formal training requires all five gate reports.")
    reports: dict[str, dict[str, Any]] = {}
    common_fields = {
        "schema_version",
        "gate",
        "passed",
        "runtime_signature_sha256",
    }
    for name in sorted(required):
        value = _load_report(Path(paths[name]), name)
        _require_fields(name, value, common_fields)
        if (
            type(value["schema_version"]) is not int
            or value["schema_version"] != GATE_SCHEMA_VERSION
            or value["gate"] != name
            or value["passed"] is not True
            or value["runtime_signature_sha256"] != signature["sha256"]
        ):
            raise RuntimeError(
                f"{name} gate report is invalid or belongs to another run."
            )
        reports[name] = value

    _validate_structure_report(config, reports["structure"])
    _validate_probability_report(config, reports["probability"])
    _validate_memory_report(config, reports["memory"])
    _validate_throughput_report(config, reports["throughput"])
    _validate_pilot_report(config, signature, reports["pilot"])

    gpu_reports = ("probability", "memory", "throughput", "pilot")
    measured_identities: dict[str, dict[str, Any]] = {}
    for name in gpu_reports:
        try:
            identity = validate_gpu_identity(reports[name]["gpu_identity"])
        except (TypeError, ValueError) as exc:
            raise RuntimeError(f"{name} gate GPU identity is invalid.") from exc
        if reports[name]["gpu_identity"] != identity:
            raise RuntimeError(f"{name} gate GPU identity is not canonical.")
        measured_identities[name] = identity
    first_identity = measured_identities[gpu_reports[0]]
    if any(measured_identities[name] != first_identity for name in gpu_reports[1:]):
        raise RuntimeError("GPU gate reports were measured on different devices.")
    if expected_gpu_identity is not None:
        try:
            expected_identity = validate_gpu_identity(expected_gpu_identity)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("Expected GPU identity is invalid.") from exc
        if first_identity != expected_identity:
            raise RuntimeError("Gate GPU identity differs from the current GPU.")
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
