"""Requirement-by-requirement verifier for a completed formal run."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from ksllm4rec_rloo.integrity import sha256_file

from . import contract
from .fingerprint import validate_runtime_signature
from .gates import load_and_validate_gate_reports
from .trainer import learning_rate_for_window


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read {label}: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must contain a JSON object.")
    return value


def _load_jsonl(path: Path, label: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise RuntimeError(f"Cannot read {label}: {path}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            value = json.loads(line)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Invalid {label} row {line_number}.") from exc
        if not isinstance(value, dict):
            raise ValueError(f"{label} row {line_number} is not an object.")
        rows.append(value)
    return rows


def _all_finite(value: Any) -> bool:
    if isinstance(value, float):
        return math.isfinite(value)
    if isinstance(value, Mapping):
        return all(_all_finite(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_finite(child) for child in value)
    return True


def validate_window_rows(
    config: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    *,
    expected_windows: int,
) -> dict[str, Any]:
    if len(rows) != int(expected_windows):
        raise RuntimeError(
            f"Expected {expected_windows} window rows, found {len(rows)}."
        )
    all_group_occurrences: list[str] = []
    total_updates = 0
    maximum_reserved = 0.0
    maximum_first_replay_delta = 0.0
    for expected_index, row in enumerate(rows):
        if row.get("window_index") != expected_index:
            raise RuntimeError("Window indices are not contiguous from zero.")
        expected_lr = learning_rate_for_window(config, expected_index)
        if not math.isclose(
            float(row.get("learning_rate")), expected_lr, rel_tol=0.0, abs_tol=1e-15
        ):
            raise RuntimeError(f"Window {expected_index} has the wrong learning rate.")
        group_ids = row.get("group_ids")
        if (
            not isinstance(group_ids, list)
            or len(group_ids) != contract.EFFECTIVE_GROUPS_PER_WINDOW
            or len(set(group_ids)) != len(group_ids)
        ):
            raise RuntimeError(f"Window {expected_index} violates K=1 group usage.")
        steps = row.get("steps")
        if not isinstance(steps, list) or len(steps) != contract.OPTIMIZER_UPDATES_PER_WINDOW:
            raise RuntimeError(f"Window {expected_index} must contain four updates.")
        step_ids = [group_id for step in steps for group_id in step.get("group_ids", [])]
        if step_ids != group_ids:
            raise RuntimeError(f"Window {expected_index} step/group ordering differs.")
        if any(len(step.get("group_ids", [])) != contract.MINIBATCH_GROUPS for step in steps):
            raise RuntimeError(f"Window {expected_index} has a non-8-group minibatch.")
        step_lrs = [float(step.get("learning_rate")) for step in steps]
        if any(
            not math.isclose(value, expected_lr, rel_tol=0.0, abs_tol=1e-15)
            for value in step_lrs
        ):
            raise RuntimeError(f"Window {expected_index} changes LR inside the window.")
        first_delta = float(steps[0].get("max_replay_logp_difference"))
        maximum_first_replay_delta = max(maximum_first_replay_delta, first_delta)
        if first_delta > float(
            config["rollout"]["canonical_replay_max_logp_difference"]
        ):
            raise RuntimeError(f"Window {expected_index} fails canonical replay parity.")
        for step in steps:
            if not (
                math.isfinite(float(step.get("ratio_min")))
                and math.isfinite(float(step.get("ratio_max")))
                and math.isfinite(float(step.get("ratio_mean")))
                and float(step.get("ratio_min")) > 0.0
            ):
                raise RuntimeError(f"Window {expected_index} contains invalid ratios.")
        if row.get("anchor_groups") != 0 or row.get("gt_injection_count") != 0:
            raise RuntimeError(f"Window {expected_index} used anchor or GT injection.")
        if row.get("k") != contract.NUM_ITERATIONS:
            raise RuntimeError(f"Window {expected_index} does not have K=1.")
        reserved = float(row.get("peak_reserved_gib"))
        maximum_reserved = max(maximum_reserved, reserved)
        if reserved > float(config["memory"]["max_reserved_gib"]):
            raise RuntimeError(f"Window {expected_index} exceeds the memory limit.")
        if not _all_finite(row):
            raise RuntimeError(f"Window {expected_index} contains NaN or Inf.")
        all_group_occurrences.extend(group_ids)
        total_updates += len(steps)
    return {
        "windows": len(rows),
        "group_occurrences": len(all_group_occurrences),
        "optimizer_updates": total_updates,
        "max_reserved_gib": maximum_reserved,
        "max_first_replay_logp_difference": maximum_first_replay_delta,
    }


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(dict(value), handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def verify_run(
    config: dict[str, Any],
    signature: dict[str, Any],
    *,
    run_dir: Path,
    gate_paths: Mapping[str, Path],
    output: Path,
) -> dict[str, Any]:
    validate_runtime_signature(signature)
    load_and_validate_gate_reports(config, signature, gate_paths)
    root = Path(run_dir)
    recorded_signature = _load_json(root / "runtime_signature.json", "run signature")
    recorded_config = _load_json(root / "resolved_config.json", "resolved config")
    if recorded_signature != signature or recorded_config != config:
        raise RuntimeError("Run directory belongs to a different runtime contract.")
    summary = _load_json(root / "run_summary.json", "run summary")
    if (
        summary.get("complete") is not True
        or summary.get("completed_windows") != contract.TOTAL_WINDOWS
        or summary.get("optimizer_update_step") != contract.TOTAL_OPTIMIZER_UPDATES
        or summary.get("anchor_groups") != 0
        or summary.get("gt_injection_count") != 0
        or summary.get("k") != contract.NUM_ITERATIONS
    ):
        raise RuntimeError("Run summary does not describe a complete approved run.")
    windows = _load_jsonl(root / "windows.jsonl", "window log")
    groups = _load_jsonl(root / "groups.jsonl", "group log")
    window_evidence = validate_window_rows(
        config, windows, expected_windows=contract.TOTAL_WINDOWS
    )
    expected_group_rows = contract.TOTAL_WINDOWS * contract.EFFECTIVE_GROUPS_PER_WINDOW
    if len(groups) != expected_group_rows:
        raise RuntimeError(
            f"Expected {expected_group_rows} group rows, found {len(groups)}."
        )
    if any(
        row.get("anchor_groups") != 0
        or row.get("gt_injection_count") != 0
        or row.get("k") != contract.NUM_ITERATIONS
        or row.get("effective") is not True
        or not _all_finite(row)
        for row in groups
    ):
        raise RuntimeError("Group log contains invalid, anchor, or non-effective rows.")
    final_adapter = root / "final-adapter"
    weights = final_adapter / "adapter_model.safetensors"
    adapter_config = final_adapter / "adapter_config.json"
    if not weights.is_file() or not adapter_config.is_file():
        raise RuntimeError("Completed run has no final adapter.")
    initial_hash = sha256_file(
        Path(config["model"]["sft_adapter"]) / "adapter_model.safetensors"
    )
    final_hash = sha256_file(weights)
    if final_hash == initial_hash:
        raise RuntimeError("Final adapter is byte-identical to the SFT starting point.")
    passed = True
    report = {
        "schema_version": 1,
        "passed": passed,
        "runtime_signature_sha256": signature["sha256"],
        "window_evidence": window_evidence,
        "group_rows": len(groups),
        "effective_epochs": contract.EFFECTIVE_EPOCHS,
        "initial_adapter_sha256": initial_hash,
        "final_adapter_sha256": final_hash,
        "parameter_changed": True,
        "anchor_groups": 0,
        "gt_injection_count": 0,
        "k": contract.NUM_ITERATIONS,
    }
    _atomic_json(output, report)
    return report


__all__ = ["validate_window_rows", "verify_run"]
