"""Strict artifact comparison for the offline two-process recovery audit."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping
from dataclasses import fields, is_dataclass

import numpy as np
import torch

from .checkpoint import load_latest_recovery
from ._infra.rloo_integrity import canonical_sha256


_BYTE_IDENTICAL_ARTIFACTS = (
    "source_epoch_plan.json",
    "source_groups.jsonl",
    "groups.jsonl",
    "windows.jsonl",
    "anchor_groups.jsonl",
    "epoch_auxiliary_flushes.jsonl",
)


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _normalized_summary(path: Path) -> dict[str, Any]:
    value = _json(path)
    value.pop("last_checkpoint", None)
    return value


def _adapter_config_semantics(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return PEFT-load semantics plus raw-byte provenance for one config.

    PEFT's adapter contract treats ``target_modules`` as a set.  Its serialization
    order therefore cannot prove that a resumed training state changed.
    """

    value = _json(path)
    target_modules = value.get("target_modules")
    if not isinstance(target_modules, list) or any(
        not isinstance(module, str) for module in target_modules
    ):
        raise ValueError(f"adapter_config target_modules must be a JSON string list: {path}")
    normalized = dict(value)
    normalized["target_modules"] = sorted(set(target_modules))
    return normalized, {
        "semantic_sha256": canonical_sha256(normalized),
        "raw_sha256": _sha256_file(path),
        "target_modules": normalized["target_modules"],
    }


def _assert_tree_equal(left: Any, right: Any, *, path: str) -> None:
    """Compare optimizer/scheduler/RNG state without trusting torch-save bytes."""

    if is_dataclass(left) or is_dataclass(right):
        if not is_dataclass(left) or not is_dataclass(right) or type(left) is not type(right):
            raise AssertionError(f"Dataclass type differs at {path}.")
        for field in fields(left):
            _assert_tree_equal(
                getattr(left, field.name),
                getattr(right, field.name),
                path=f"{path}.{field.name}",
            )
        return
    if isinstance(left, torch.Tensor) or isinstance(right, torch.Tensor):
        if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
            raise AssertionError(f"Tensor type differs at {path}.")
        if left.dtype != right.dtype or tuple(left.shape) != tuple(right.shape):
            raise AssertionError(f"Tensor metadata differs at {path}.")
        if not torch.equal(left.cpu(), right.cpu()):
            raise AssertionError(f"Tensor values differ at {path}.")
        return
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        if not isinstance(left, np.ndarray) or not isinstance(right, np.ndarray):
            raise AssertionError(f"NumPy type differs at {path}.")
        if left.dtype != right.dtype or left.shape != right.shape or not np.array_equal(left, right):
            raise AssertionError(f"NumPy values differ at {path}.")
        return
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping):
            raise AssertionError(f"Mapping type differs at {path}.")
        if set(left) != set(right):
            raise AssertionError(f"Mapping keys differ at {path}.")
        for key in sorted(left, key=str):
            _assert_tree_equal(left[key], right[key], path=f"{path}[{key!r}]")
        return
    if isinstance(left, (tuple, list)) or isinstance(right, (tuple, list)):
        if type(left) is not type(right) or len(left) != len(right):
            raise AssertionError(f"Sequence shape differs at {path}.")
        for index, (left_value, right_value) in enumerate(zip(left, right, strict=True)):
            _assert_tree_equal(left_value, right_value, path=f"{path}[{index}]")
        return
    if left != right:
        raise AssertionError(f"Values differ at {path}: {left!r} != {right!r}.")


def compare_recovery_trajectories(
    continuous_dir: Path,
    recovered_dir: Path,
    contract_signature: Mapping[str, Any],
) -> dict[str, Any]:
    """Prove two trajectories produce identical committed training state."""

    continuous = Path(continuous_dir).resolve()
    recovered = Path(recovered_dir).resolve()
    artifact_hashes: dict[str, str] = {}
    for relative in _BYTE_IDENTICAL_ARTIFACTS:
        left = continuous / relative
        right = recovered / relative
        if not left.is_file() or not right.is_file():
            raise FileNotFoundError(f"Recovery audit artifact missing: {relative}")
        left_sha = _sha256_file(left)
        right_sha = _sha256_file(right)
        if left_sha != right_sha:
            raise AssertionError(f"Byte-identical artifact differs: {relative}")
        artifact_hashes[relative] = left_sha
    if _normalized_summary(continuous / "run_summary.json") != _normalized_summary(
        recovered / "run_summary.json"
    ):
        raise AssertionError("Run summaries differ after removing output-local checkpoint paths.")
    left_loaded = load_latest_recovery(continuous / "recovery", contract_signature)
    right_loaded = load_latest_recovery(recovered / "recovery", contract_signature)
    if left_loaded is None or right_loaded is None:
        raise RuntimeError("Recovery audit requires one committed checkpoint per trajectory.")
    left_checkpoint, left_state, left_training = left_loaded
    right_checkpoint, right_state, right_training = right_loaded
    _assert_tree_equal(left_state, right_state, path="recovery_state")
    _assert_tree_equal(left_training, right_training, path="training_state")
    left_model = left_checkpoint / "adapter_model.safetensors"
    right_model = right_checkpoint / "adapter_model.safetensors"
    if not left_model.is_file() or not right_model.is_file():
        raise FileNotFoundError("Recovery policy artifact missing: adapter_model.safetensors")
    left_model_sha = _sha256_file(left_model)
    right_model_sha = _sha256_file(right_model)
    if left_model_sha != right_model_sha:
        raise AssertionError("Recovered policy artifact differs: adapter_model.safetensors")

    left_config = left_checkpoint / "adapter_config.json"
    right_config = right_checkpoint / "adapter_config.json"
    if not left_config.is_file() or not right_config.is_file():
        raise FileNotFoundError("Recovery policy artifact missing: adapter_config.json")
    left_config_semantics, left_config_record = _adapter_config_semantics(left_config)
    right_config_semantics, right_config_record = _adapter_config_semantics(right_config)
    if left_config_semantics != right_config_semantics:
        raise AssertionError("Recovered adapter configuration differs semantically.")
    return {
        "passed": True,
        "continuous_checkpoint": left_checkpoint.name,
        "recovered_checkpoint": right_checkpoint.name,
        "completed_source_blocks": left_state.next_source_block,
        "completed_policy_steps": left_state.completed_policy_steps,
        "total_optimizer_steps": left_state.total_optimizer_steps,
        "artifact_sha256": artifact_hashes,
        "adapter_sha256": {"adapter_model.safetensors": left_model_sha},
        "adapter_config": {
            "semantic_sha256": left_config_record["semantic_sha256"],
            "target_modules": left_config_record["target_modules"],
            "raw_byte_identical": left_config_record["raw_sha256"]
            == right_config_record["raw_sha256"],
            "continuous_raw_sha256": left_config_record["raw_sha256"],
            "recovered_raw_sha256": right_config_record["raw_sha256"],
        },
    }


__all__ = ["compare_recovery_trajectories"]
