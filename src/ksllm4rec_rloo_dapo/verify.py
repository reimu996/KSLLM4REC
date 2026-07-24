"""Requirement-by-requirement verifier for a completed formal run."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from safetensors import safe_open
import torch

from ksllm4rec_rloo.integrity import sha256_file
from ksllm4rec_rloo.modeling import load_policy_model, validate_adapter_contract

from . import contract
from .checkpoint import restore_training_state, validate_recovery_checkpoint
from .fingerprint import validate_runtime_signature
from .gates import load_and_validate_gate_reports
from .trainer import WindowLRScheduler, build_optimizer, learning_rate_for_window


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
        if (
            not isinstance(steps, list)
            or len(steps) != contract.OPTIMIZER_UPDATES_PER_WINDOW
        ):
            raise RuntimeError(f"Window {expected_index} must contain four updates.")
        step_ids = [
            group_id for step in steps for group_id in step.get("group_ids", [])
        ]
        if step_ids != group_ids:
            raise RuntimeError(f"Window {expected_index} step/group ordering differs.")
        if any(
            len(step.get("group_ids", [])) != contract.MINIBATCH_GROUPS
            for step in steps
        ):
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
            raise RuntimeError(
                f"Window {expected_index} fails canonical replay parity."
            )
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


def validate_group_rows(
    windows: Sequence[Mapping[str, Any]],
    groups: Sequence[Mapping[str, Any]],
) -> dict[str, int]:
    expected_rows = len(windows) * contract.EFFECTIVE_GROUPS_PER_WINDOW
    if len(groups) != expected_rows:
        raise RuntimeError(f"Expected {expected_rows} group rows, found {len(groups)}.")
    if any(
        row.get("anchor_groups") != 0
        or row.get("gt_injection_count") != 0
        or row.get("k") != contract.NUM_ITERATIONS
        or row.get("effective") is not True
        or not _all_finite(row)
        for row in groups
    ):
        raise RuntimeError("Group log contains invalid, anchor, or non-effective rows.")
    for window_index, window in enumerate(windows):
        start = window_index * contract.EFFECTIVE_GROUPS_PER_WINDOW
        window_groups = groups[start : start + contract.EFFECTIVE_GROUPS_PER_WINDOW]
        group_ids = [row.get("group_id") for row in window_groups]
        if (
            any(row.get("window_index") != window_index for row in window_groups)
            or any(
                not isinstance(group_id, str) or not group_id for group_id in group_ids
            )
            or len(set(group_ids)) != contract.EFFECTIVE_GROUPS_PER_WINDOW
            or set(group_ids) != set(window["group_ids"])
        ):
            raise RuntimeError(f"Group log rows do not match window {window_index}.")
        for row in window_groups:
            if (
                row.get("candidate_indices") != list(range(contract.GROUP_SIZE))
                or len(row.get("candidate_sids", [])) != contract.GROUP_SIZE
                or len(row.get("candidate_token_ids", [])) != contract.GROUP_SIZE
                or len(row.get("rewards", [])) != contract.GROUP_SIZE
                or len(row.get("reward_tiers", [])) != contract.GROUP_SIZE
                or len(row.get("advantages", [])) != contract.GROUP_SIZE
            ):
                raise RuntimeError(
                    f"Group log candidate payload is invalid in window {window_index}."
                )
    return {"group_rows": len(groups)}


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


def validate_final_adapter(path: Path, reference_path: Path) -> dict[str, Any]:
    """Parse the final LoRA and prove its exact structure and finite values."""

    root = Path(path)
    adapter = validate_adapter_contract(root)
    if adapter.source_dropout != contract.LORA_DROPOUT:
        raise RuntimeError("Final adapter must serialize LoRA dropout 0.0.")
    weights = root / "adapter_model.safetensors"
    reference_weights = Path(reference_path) / "adapter_model.safetensors"
    tensor_count = 0
    parameter_count = 0
    changed_tensor_count = 0
    try:
        with (
            safe_open(str(weights), framework="pt", device="cpu") as handle,
            safe_open(
                str(reference_weights), framework="pt", device="cpu"
            ) as reference,
        ):
            keys = list(handle.keys())
            reference_keys = list(reference.keys())
            if keys != reference_keys:
                raise RuntimeError(
                    "Final adapter tensor keys differ from the SFT adapter."
                )
            tensor_count = len(keys)
            for key in keys:
                tensor = handle.get_tensor(key)
                reference_tensor = reference.get_tensor(key)
                if (
                    tensor.shape != reference_tensor.shape
                    or tensor.dtype != reference_tensor.dtype
                ):
                    raise RuntimeError(
                        f"Final adapter tensor schema differs from SFT: {key}."
                    )
                if not tensor.is_floating_point() or not bool(
                    torch.isfinite(tensor).all()
                ):
                    raise RuntimeError(
                        f"Final adapter tensor is non-floating or non-finite: {key}."
                    )
                changed_tensor_count += int(not torch.equal(tensor, reference_tensor))
                parameter_count += tensor.numel()
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("Final adapter safetensors file is unreadable.") from exc
    if tensor_count != contract.EXPECTED_LORA_TENSOR_COUNT:
        raise RuntimeError(
            f"Final adapter has {tensor_count} tensors, expected "
            f"{contract.EXPECTED_LORA_TENSOR_COUNT}."
        )
    if parameter_count != contract.EXPECTED_LORA_PARAMETER_COUNT:
        raise RuntimeError(
            f"Final adapter has {parameter_count} parameters, expected "
            f"{contract.EXPECTED_LORA_PARAMETER_COUNT}."
        )
    if changed_tensor_count <= 0:
        raise RuntimeError("Final adapter tensor values did not change from SFT.")
    return {
        "tensor_count": tensor_count,
        "parameter_count": parameter_count,
        "rank": adapter.rank,
        "alpha": adapter.alpha,
        "dropout": adapter.source_dropout,
        "changed_tensor_count": changed_tensor_count,
    }


def _all_training_state_finite(value: Any) -> bool:
    if isinstance(value, torch.Tensor):
        return not value.is_floating_point() or bool(torch.isfinite(value).all())
    if isinstance(value, Mapping):
        return all(_all_training_state_finite(child) for child in value.values())
    if isinstance(value, (list, tuple)):
        return all(_all_training_state_finite(child) for child in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def _normalized_adapter_config(path: Path) -> dict[str, Any]:
    value = _load_json(Path(path), "adapter config")
    targets = value.get("target_modules")
    if not isinstance(targets, list) or any(
        not isinstance(target, str) for target in targets
    ):
        raise RuntimeError("Adapter config target_modules is invalid.")
    normalized = dict(value)
    normalized["target_modules"] = sorted(targets)
    return normalized


def verify_run(
    config: dict[str, Any],
    signature: dict[str, Any],
    *,
    run_dir: Path,
    gate_paths: Mapping[str, Path],
    output: Path,
    device: str = contract.EXECUTION_DEVICE,
    restore_recovery: bool = True,
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
    group_evidence = validate_group_rows(windows, groups)
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
    adapter_evidence = validate_final_adapter(
        final_adapter, Path(config["model"]["sft_adapter"])
    )
    epoch_evidence: list[dict[str, Any]] = []
    final_boundary_cursor = None
    final_boundary_state = None
    for epoch_number in range(1, contract.EFFECTIVE_EPOCHS + 1):
        epoch_adapter = root / f"epoch-{epoch_number:02d}-adapter"
        epoch_weights = epoch_adapter / "adapter_model.safetensors"
        epoch_config = epoch_adapter / "adapter_config.json"
        if not epoch_weights.is_file() or not epoch_config.is_file():
            raise RuntimeError(
                f"Completed run is missing epoch {epoch_number} adapter."
            )
        boundary_window = epoch_number * contract.WINDOWS_PER_EFFECTIVE_EPOCH
        boundary_updates = boundary_window * contract.OPTIMIZER_UPDATES_PER_WINDOW
        boundary_checkpoint = (
            root
            / "recovery"
            / (f"checkpoint-window-{boundary_window:06d}-update-{boundary_updates:06d}")
        )
        if epoch_number == contract.EFFECTIVE_EPOCHS:
            if sha256_file(epoch_weights) != final_hash:
                raise RuntimeError("Epoch-02 adapter differs from the final adapter.")
            epoch_adapter_evidence = adapter_evidence
        else:
            epoch_adapter_evidence = validate_final_adapter(
                epoch_adapter, Path(config["model"]["sft_adapter"])
            )
        boundary_cursor, boundary_state = validate_recovery_checkpoint(
            boundary_checkpoint, signature
        )
        if (
            boundary_cursor.next_window_index != boundary_window
            or boundary_cursor.optimizer_update_step != boundary_updates
            or not _all_training_state_finite(boundary_state)
            or sha256_file(boundary_checkpoint / "adapter_model.safetensors")
            != sha256_file(epoch_weights)
            or _normalized_adapter_config(boundary_checkpoint / "adapter_config.json")
            != _normalized_adapter_config(epoch_config)
        ):
            raise RuntimeError(f"Epoch {epoch_number} adapter/checkpoint is invalid.")
        epoch_evidence.append(
            {
                "epoch": epoch_number,
                "window": boundary_window,
                "optimizer_updates": boundary_updates,
                "adapter_sha256": sha256_file(epoch_weights),
                "adapter": epoch_adapter_evidence,
            }
        )
        if epoch_number == contract.EFFECTIVE_EPOCHS:
            final_boundary_cursor = boundary_cursor
            final_boundary_state = boundary_state
        else:
            del boundary_state

    checkpoint_value = summary.get("last_checkpoint")
    checkpoint_relative = (
        Path(checkpoint_value) if isinstance(checkpoint_value, str) else None
    )
    expected_checkpoint_name = (
        f"checkpoint-window-{contract.TOTAL_WINDOWS:06d}"
        f"-update-{contract.TOTAL_OPTIMIZER_UPDATES:06d}"
    )
    if (
        checkpoint_relative is None
        or checkpoint_relative.is_absolute()
        or ".." in checkpoint_relative.parts
        or checkpoint_relative.name != expected_checkpoint_name
    ):
        raise RuntimeError(
            "Run summary does not identify the final recovery checkpoint."
        )
    checkpoint = (root / checkpoint_relative).resolve()
    try:
        checkpoint.relative_to(root.resolve())
    except ValueError as exc:
        raise RuntimeError(
            "Final recovery checkpoint escapes the run directory."
        ) from exc
    expected_checkpoint = (root / "recovery" / expected_checkpoint_name).resolve()
    if checkpoint != expected_checkpoint:
        raise RuntimeError("Final recovery checkpoint is outside recovery/.")
    if final_boundary_cursor is None or final_boundary_state is None:
        raise RuntimeError("Final epoch recovery state was not validated.")
    cursor, training_state = final_boundary_cursor, final_boundary_state
    if (
        cursor.next_window_index != contract.TOTAL_WINDOWS
        or cursor.optimizer_update_step != contract.TOTAL_OPTIMIZER_UPDATES
        or training_state.get("scheduler")
        != {"completed_windows": contract.TOTAL_WINDOWS}
        or not isinstance(training_state.get("optimizer"), dict)
        or not training_state["optimizer"]
    ):
        raise RuntimeError("Final recovery state is incomplete or at the wrong step.")
    if not _all_training_state_finite(training_state):
        raise RuntimeError("Final recovery training state contains NaN or Inf.")
    if restore_recovery:
        bundle = load_policy_model(
            config, device=device, policy_adapter_path=checkpoint
        )
        optimizer = build_optimizer(bundle, config)
        scheduler = WindowLRScheduler(optimizer, config)
        restore_training_state(optimizer, scheduler, training_state)
        if scheduler.completed_windows != contract.TOTAL_WINDOWS or not (
            _all_training_state_finite(optimizer.state_dict())
        ):
            raise RuntimeError("Final recovery state cannot be restored exactly.")
    checkpoint_weights = checkpoint / "adapter_model.safetensors"
    checkpoint_config = checkpoint / "adapter_config.json"
    if sha256_file(checkpoint_weights) != final_hash or _normalized_adapter_config(
        checkpoint_config
    ) != _normalized_adapter_config(adapter_config):
        raise RuntimeError("Final adapter differs from the final recovery checkpoint.")
    passed = True
    report = {
        "schema_version": 1,
        "passed": passed,
        "runtime_signature_sha256": signature["sha256"],
        "window_evidence": window_evidence,
        **group_evidence,
        "effective_epochs": contract.EFFECTIVE_EPOCHS,
        "initial_adapter_sha256": initial_hash,
        "final_adapter_sha256": final_hash,
        "final_adapter": adapter_evidence,
        "epoch_adapters": epoch_evidence,
        "final_checkpoint": checkpoint_relative.as_posix(),
        "recovery_restore_passed": bool(restore_recovery),
        "parameter_changed": True,
        "anchor_groups": 0,
        "gt_injection_count": 0,
        "k": contract.NUM_ITERATIONS,
    }
    _atomic_json(output, report)
    return report


__all__ = [
    "validate_final_adapter",
    "validate_group_rows",
    "validate_window_rows",
    "verify_run",
]
