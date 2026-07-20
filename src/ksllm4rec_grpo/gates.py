"""GPU memory, gradient-signal, and wall-time gates for GRPO V3.1."""

from __future__ import annotations

import gc
import json
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from .constraint import RecommendationGrammar
from .contract import expected_trie_leaf_count
from .data import iter_groups
from .fingerprint import runtime_signature
from .modeling import load_dual_adapter_model
from .prompt import encode_prompt
from .rollout import rollout_group
from .trainer import _optimizer_and_scheduler, process_group, run_training
from .trie import SidPrefixTrie


def gpu_status(device: str = "cuda:0") -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    torch_device = torch.device(device)
    index = torch_device.index or 0
    free, total = torch.cuda.mem_get_info(index)
    properties = torch.cuda.get_device_properties(index)
    return {
        "device": str(torch_device),
        "name": properties.name,
        "free_gib": free / 1024**3,
        "total_gib": total / 1024**3,
    }


def _longest_group(config: dict[str, Any], groups_path: Path):
    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["tokenizer"],
        local_files_only=True,
        trust_remote_code=True,
    )
    cutoff = int(config["data"]["cutoff_len"])
    selected = None
    selected_ids = None
    for group in iter_groups(groups_path):
        ids = encode_prompt(tokenizer, group.system, group.prompt, cutoff_len=cutoff)
        if selected_ids is None or len(ids) > len(selected_ids):
            selected = group
            selected_ids = ids
    if selected is None or selected_ids is None:
        raise RuntimeError("No recommendation groups were loaded.")
    return selected, selected_ids


def _clean_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def _attempt_rollout(
    config: dict[str, Any],
    group,
    prompt_ids,
    trie_dir: Path,
    chunk: int,
    device: str,
) -> dict[str, Any]:
    _clean_cuda()
    started = time.monotonic()
    bundle = None
    try:
        bundle = load_dual_adapter_model(config, device=device)
        trie = SidPrefixTrie.load(
            trie_dir, expected_leaf_count=expected_trie_leaf_count(config)
        )
        grammar = RecommendationGrammar(bundle.tokenizer, trie)
        with bundle.use_policy():
            candidates = rollout_group(
                bundle.model,
                prompt_ids,
                grammar,
                group_id=group.group_id,
                epoch_index=1,
                chunk_size=chunk,
                max_completion_length=int(config["rollout"]["max_completion_length"]),
                device=device,
            )
        return {
            "success": True,
            "phase": "rollout",
            "chunk": chunk,
            "candidates": len(candidates),
            "seconds": time.monotonic() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        }
    except BaseException as exc:
        if not (
            isinstance(exc, torch.cuda.OutOfMemoryError)
            or "out of memory" in str(exc).lower()
        ):
            raise
        return {
            "success": False,
            "phase": "rollout",
            "chunk": chunk,
            "error": type(exc).__name__,
            "seconds": time.monotonic() - started,
        }
    finally:
        del bundle
        _clean_cuda()


def _attempt_loss(
    config: dict[str, Any],
    group,
    prompt_ids,
    trie_dir: Path,
    rollout_chunk: int,
    loss_chunk: int,
    device: str,
) -> dict[str, Any]:
    _clean_cuda()
    started = time.monotonic()
    bundle = None
    try:
        bundle = load_dual_adapter_model(config, device=device)
        trie = SidPrefixTrie.load(
            trie_dir, expected_leaf_count=expected_trie_leaf_count(config)
        )
        grammar = RecommendationGrammar(bundle.tokenizer, trie)
        optimizer, scheduler, parameters = _optimizer_and_scheduler(bundle, config)
        # First step allocates Adam states; the second group measures steady-state memory.
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            process_group(
                bundle,
                group,
                grammar,
                prompt_ids,
                epoch_index=1,
                rollout_chunk=rollout_chunk,
                loss_chunk=loss_chunk,
                max_completion_length=int(config["rollout"]["max_completion_length"]),
                window_groups=1,
                clip_epsilon=float(config["loss"]["clip_epsilon"]),
                beta=float(config["loss"]["reference_beta"]),
                std_epsilon=float(config["reward"]["std_epsilon"]),
                device=torch.device(device),
            )
            torch.nn.utils.clip_grad_norm_(
                parameters, float(config["train"]["max_grad_norm"])
            )
            optimizer.step()
            scheduler.step()
        return {
            "success": True,
            "phase": "loss",
            "rollout_chunk": rollout_chunk,
            "chunk": loss_chunk,
            "seconds": time.monotonic() - started,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
        }
    except BaseException as exc:
        is_oom = isinstance(exc, torch.cuda.OutOfMemoryError) or (
            "out of memory" in str(exc).lower()
        )
        is_on_policy_mismatch = isinstance(exc, RuntimeError) and (
            "Strict on-policy log-probability mismatch" in str(exc)
        )
        if not (is_oom or is_on_policy_mismatch):
            raise
        return {
            "success": False,
            "phase": "loss",
            "rollout_chunk": rollout_chunk,
            "chunk": loss_chunk,
            "error": type(exc).__name__,
            "failure_kind": (
                "on_policy_mismatch" if is_on_policy_mismatch else "out_of_memory"
            ),
            "message": str(exc),
            "seconds": time.monotonic() - started,
        }
    finally:
        del bundle
        _clean_cuda()


def run_memory_gate(
    config: dict[str, Any],
    *,
    groups_path: Path,
    trie_dir: Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    status = gpu_status(device)
    minimum_free = float(config["memory"]["min_free_gib"])
    if status["free_gib"] < minimum_free:
        raise RuntimeError(
            f"GPU free memory {status['free_gib']:.3f} GiB is below "
            f"the required {minimum_free:.3f} GiB."
        )
    group, prompt_ids = _longest_group(config, groups_path)
    maximum_reserved = float(config["memory"]["max_reserved_gib"])
    device_index = torch.device(device).index or 0
    total_bytes = torch.cuda.get_device_properties(device_index).total_memory
    allocator_fraction = min(maximum_reserved * 1024**3 / total_bytes, 1.0)
    torch.cuda.set_per_process_memory_fraction(allocator_fraction, device_index)
    attempts: list[dict[str, Any]] = []
    selected_chunk = None
    try:
        for chunk in config["memory"]["chunk_candidates"]:
            rollout_result = _attempt_rollout(
                config, group, prompt_ids, trie_dir, int(chunk), device
            )
            attempts.append(rollout_result)
            if not (
                rollout_result.get("success")
                and rollout_result["peak_reserved_gib"] <= maximum_reserved
            ):
                continue
            loss_result = _attempt_loss(
                config,
                group,
                prompt_ids,
                trie_dir,
                int(chunk),
                int(chunk),
                device,
            )
            attempts.append(loss_result)
            if (
                loss_result.get("success")
                and loss_result["peak_reserved_gib"] <= maximum_reserved
            ):
                selected_chunk = int(chunk)
                break
    finally:
        torch.cuda.set_per_process_memory_fraction(1.0, device_index)
    if selected_chunk is None:
        raise RuntimeError(f"No unified on-policy chunk passed: {attempts}")
    return {
        "runtime_signature": runtime_signature(config, groups_path, trie_dir),
        "gpu": status,
        "longest_group_id": group.group_id,
        "longest_prompt_tokens": len(prompt_ids),
        "max_reserved_gib": maximum_reserved,
        "allocator_fraction": allocator_fraction,
        "rollout_chunk": selected_chunk,
        "loss_chunk": selected_chunk,
        "attempts": attempts,
    }


def run_signal_gate(
    config: dict[str, Any],
    *,
    groups_path: Path,
    trie_dir: Path,
    output_dir: Path,
    rollout_chunk: int,
    loss_chunk: int,
    device: str = "cuda:0",
) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    result = run_training(
        config,
        groups_path=groups_path,
        trie_dir=trie_dir,
        output_dir=output_dir,
        rollout_chunk=rollout_chunk,
        loss_chunk=loss_chunk,
        max_groups=int(config["gates"]["signal_groups"]),
        resume=False,
        save_recovery=False,
        save_epochs=False,
        device=device,
    )
    result["runtime_signature"] = runtime_signature(config, groups_path, trie_dir)
    passed = (
        result["groups_run"] == int(config["gates"]["signal_groups"])
        and result["signal_groups"] > 0
        and result["gradient_norm_max"] > 0.0
        and result["parameter_max_change_this_invocation"] > 0.0
        and result["loss_mean"] is not None
        and torch.isfinite(torch.tensor(result["loss_mean"])).item()
    )
    result["passed"] = bool(passed)
    if not passed:
        raise RuntimeError(f"Signal gate failed: {result}")
    return result


def run_timing_gate(
    config: dict[str, Any],
    *,
    groups_path: Path,
    trie_dir: Path,
    output_dir: Path,
    rollout_chunk: int,
    loss_chunk: int,
    device: str = "cuda:0",
) -> dict[str, Any]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    result = run_training(
        config,
        groups_path=groups_path,
        trie_dir=trie_dir,
        output_dir=output_dir,
        rollout_chunk=rollout_chunk,
        loss_chunk=loss_chunk,
        max_groups=int(config["gates"]["timing_groups"]),
        resume=False,
        save_recovery=False,
        save_epochs=False,
        device=device,
    )
    result["runtime_signature"] = runtime_signature(config, groups_path, trie_dir)
    total_groups = int(config["data"]["groups"]) * int(config["train"]["epochs"])
    projected_hours = (
        result["train_seconds"] / result["groups_run"] * total_groups / 3600
    )
    result["projected_two_epoch_hours"] = projected_hours
    result["maximum_projected_hours"] = float(config["gates"]["max_projected_hours"])
    result["passed"] = projected_hours <= result["maximum_projected_hours"]
    if not result["passed"]:
        raise RuntimeError(f"Timing gate failed: {result}")
    return result


def _load_report(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Gate report must be a JSON object: {path}")
    return value


def _validate_memory_report(
    config: dict[str, Any],
    report: dict[str, Any],
    signature: dict[str, Any],
) -> tuple[int, int]:
    if report.get("runtime_signature") != signature:
        raise RuntimeError("Memory gate belongs to a different runtime contract.")
    rollout_chunk = int(report["rollout_chunk"])
    loss_chunk = int(report["loss_chunk"])
    allowed = tuple(int(value) for value in config["memory"]["chunk_candidates"])
    if rollout_chunk != loss_chunk or rollout_chunk not in allowed:
        raise RuntimeError("Memory gate did not select one approved unified chunk.")
    maximum_reserved = float(config["memory"]["max_reserved_gib"])
    if not any(
        attempt.get("success")
        and attempt.get("phase") == "loss"
        and int(attempt.get("rollout_chunk", -1)) == rollout_chunk
        and int(attempt.get("chunk", -1)) == loss_chunk
        and float(attempt.get("peak_reserved_gib", float("inf"))) <= maximum_reserved
        for attempt in report.get("attempts", [])
    ):
        raise RuntimeError("Memory gate has no passing steady-state loss attempt.")
    return rollout_chunk, loss_chunk


def load_memory_gate(
    path: Path,
    *,
    config: dict[str, Any],
    groups_path: Path,
    trie_dir: Path,
) -> tuple[int, int]:
    signature = runtime_signature(config, groups_path, trie_dir)
    return _validate_memory_report(config, _load_report(path), signature)


def load_training_gates(
    *,
    config: dict[str, Any],
    groups_path: Path,
    trie_dir: Path,
    memory_path: Path,
    signal_path: Path,
    timing_path: Path,
) -> tuple[int, int]:
    signature = runtime_signature(config, groups_path, trie_dir)
    rollout_chunk, loss_chunk = _validate_memory_report(
        config, _load_report(memory_path), signature
    )
    signal = _load_report(signal_path)
    signal_state = signal.get("state", {})
    if (
        signal.get("runtime_signature") != signature
        or signal.get("passed") is not True
        or int(signal.get("groups_run", -1)) != int(config["gates"]["signal_groups"])
        or float(signal.get("gradient_norm_max", 0.0)) <= 0.0
        or float(signal.get("parameter_max_change_this_invocation", 0.0)) <= 0.0
        or int(signal_state.get("rollout_chunk", -1)) != rollout_chunk
        or int(signal_state.get("loss_chunk", -1)) != loss_chunk
    ):
        raise RuntimeError("Signal gate did not pass this runtime contract.")
    timing = _load_report(timing_path)
    timing_state = timing.get("state", {})
    if (
        timing.get("runtime_signature") != signature
        or timing.get("passed") is not True
        or int(timing.get("groups_run", -1)) != int(config["gates"]["timing_groups"])
        or float(timing.get("projected_two_epoch_hours", float("inf")))
        > float(config["gates"]["max_projected_hours"])
        or int(timing_state.get("rollout_chunk", -1)) != rollout_chunk
        or int(timing_state.get("loss_chunk", -1)) != loss_chunk
    ):
        raise RuntimeError("Timing gate did not pass this runtime contract.")
    return rollout_chunk, loss_chunk
