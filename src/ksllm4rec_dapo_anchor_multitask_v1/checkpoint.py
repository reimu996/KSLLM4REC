"""Atomic recovery for DAPO-Anchor-Multitask V1 optimization windows."""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ._infra.rloo_integrity import (
    canonical_sha256,
    sha256_file,
    snapshot_directory,
    verify_directory_snapshot,
)


CHECKPOINT_SCHEMA_VERSION = 2
_MANIFEST = "manifest.json"
_MANIFEST_SHA = "manifest.sha256"
_CHECKPOINT_RE = re.compile(
    r"checkpoint-block-(\d+)-window-(\d+)-update-(\d+)"
)


def _non_negative_integer(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label} must be a non-negative integer.")
    return value


@dataclass(frozen=True)
class PendingBuffer:
    """Opaque not-yet-optimized data and the policy step that sampled it."""

    snapshot_policy_step: int
    payload: Any

    def __post_init__(self) -> None:
        _non_negative_integer(
            self.snapshot_policy_step, "PendingBuffer.snapshot_policy_step"
        )
        if self.payload is None:
            raise ValueError("PendingBuffer.payload must not be None.")


@dataclass(frozen=True)
class RecoveryState:
    """Progress committed together with one exact policy/optimizer snapshot."""

    next_source_block: int
    optimization_window_index: int
    completed_policy_steps: int
    total_optimizer_steps: int
    final_auxiliary_flush_steps: int
    pending_buffer: PendingBuffer | None
    last_rl_reference_grad_norm: float | None
    source_log_rows: int
    group_log_rows: int
    window_log_rows: int
    final_auxiliary_flush: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        for name in (
            "next_source_block",
            "optimization_window_index",
            "completed_policy_steps",
            "total_optimizer_steps",
            "final_auxiliary_flush_steps",
            "source_log_rows",
            "group_log_rows",
            "window_log_rows",
        ):
            _non_negative_integer(getattr(self, name), f"RecoveryState.{name}")
        if self.final_auxiliary_flush_steps > 1:
            raise ValueError("At most one final auxiliary flush step is allowed.")
        if self.total_optimizer_steps != (
            self.completed_policy_steps + self.final_auxiliary_flush_steps
        ):
            raise ValueError(
                "total_optimizer_steps must equal completed_policy_steps plus "
                "final_auxiliary_flush_steps."
            )
        final_flush = self.final_auxiliary_flush
        if final_flush is not None:
            if not isinstance(final_flush, Mapping):
                raise TypeError("final_auxiliary_flush must be a mapping or None.")
            optimizer_steps = _non_negative_integer(
                final_flush.get("optimizer_steps"),
                "RecoveryState.final_auxiliary_flush.optimizer_steps",
            )
            if optimizer_steps > 1:
                raise ValueError("The final auxiliary flush permits at most one step.")
            if optimizer_steps != self.final_auxiliary_flush_steps:
                raise ValueError(
                    "Final auxiliary audit and optimizer-step counters differ."
                )
        elif self.final_auxiliary_flush_steps:
            raise ValueError(
                "A committed final auxiliary optimizer step needs its audit record."
            )
        if self.pending_buffer is not None:
            if not isinstance(self.pending_buffer, PendingBuffer):
                raise TypeError("pending_buffer must be PendingBuffer or None.")
            if (
                self.pending_buffer.snapshot_policy_step
                != self.completed_policy_steps
            ):
                raise ValueError(
                    "Pending buffer snapshot_policy_step must equal the current "
                    "completed_policy_steps."
                )
        value = self.last_rl_reference_grad_norm
        if value is not None:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(
                    "last_rl_reference_grad_norm must be a number or None."
                )
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(
                    "last_rl_reference_grad_norm must be finite and non-negative."
                )


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    if set(state) != {"python", "numpy", "torch_cpu", "torch_cuda"}:
        raise ValueError("Recovery RNG state has an invalid schema.")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        cuda_states = state["torch_cuda"]
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("Recovery CUDA RNG state has a different device count.")
        torch.cuda.set_rng_state_all(cuda_states)


def _validate_signature(signature: Mapping[str, Any]) -> dict[str, Any]:
    value = dict(signature)
    if (
        set(value) != {"sha256", "inputs"}
        or value["sha256"] != canonical_sha256(value["inputs"])
    ):
        raise ValueError("Runtime contract signature is invalid.")
    return value


def _progress_value(state: RecoveryState) -> dict[str, Any]:
    pending = state.pending_buffer
    return {
        "next_source_block": state.next_source_block,
        "optimization_window_index": state.optimization_window_index,
        "completed_policy_steps": state.completed_policy_steps,
        "total_optimizer_steps": state.total_optimizer_steps,
        "final_auxiliary_flush_steps": state.final_auxiliary_flush_steps,
        "pending_buffer": (
            None
            if pending is None
            else {"snapshot_policy_step": pending.snapshot_policy_step}
        ),
        "last_rl_reference_grad_norm": state.last_rl_reference_grad_norm,
        "source_log_rows": state.source_log_rows,
        "group_log_rows": state.group_log_rows,
        "window_log_rows": state.window_log_rows,
        "final_auxiliary_flush": (
            None
            if state.final_auxiliary_flush is None
            else dict(state.final_auxiliary_flush)
        ),
    }


def _state_from_values(progress: Mapping[str, Any], payload: Any) -> RecoveryState:
    expected = {
        "next_source_block",
        "optimization_window_index",
        "completed_policy_steps",
        "total_optimizer_steps",
        "final_auxiliary_flush_steps",
        "pending_buffer",
        "last_rl_reference_grad_norm",
        "source_log_rows",
        "group_log_rows",
        "window_log_rows",
        "final_auxiliary_flush",
    }
    if set(progress) != expected:
        raise ValueError("Recovery progress has an invalid schema.")
    pending_metadata = progress["pending_buffer"]
    if pending_metadata is None:
        if payload is not None:
            raise RuntimeError("Recovery has a pending payload without metadata.")
        pending = None
    else:
        if (
            not isinstance(pending_metadata, Mapping)
            or set(pending_metadata) != {"snapshot_policy_step"}
        ):
            raise ValueError("Recovery pending-buffer metadata has an invalid schema.")
        if payload is None:
            raise RuntimeError("Recovery pending-buffer payload is missing.")
        pending = PendingBuffer(
            snapshot_policy_step=pending_metadata["snapshot_policy_step"],
            payload=payload,
        )
    return RecoveryState(
        next_source_block=progress["next_source_block"],
        optimization_window_index=progress["optimization_window_index"],
        completed_policy_steps=progress["completed_policy_steps"],
        total_optimizer_steps=progress["total_optimizer_steps"],
        final_auxiliary_flush_steps=progress["final_auxiliary_flush_steps"],
        pending_buffer=pending,
        last_rl_reference_grad_norm=progress["last_rl_reference_grad_norm"],
        source_log_rows=progress["source_log_rows"],
        group_log_rows=progress["group_log_rows"],
        window_log_rows=progress["window_log_rows"],
        final_auxiliary_flush=progress["final_auxiliary_flush"],
    )


def _checkpoint_name(state: RecoveryState) -> str:
    return (
        f"checkpoint-block-{state.next_source_block:06d}"
        f"-window-{state.optimization_window_index:06d}"
        f"-update-{state.total_optimizer_steps:06d}"
    )


def _parse_checkpoint(path: Path) -> tuple[int, int, int] | None:
    match = _CHECKPOINT_RE.fullmatch(path.name)
    if not path.is_dir() or match is None:
        return None
    return tuple(int(match.group(index)) for index in (1, 2, 3))


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def save_recovery_checkpoint(
    bundle: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: RecoveryState,
    recovery_root: Path,
    contract_signature: Mapping[str, Any],
) -> Path:
    """Commit policy and all continuation state as one immutable directory."""

    if not isinstance(state, RecoveryState):
        raise TypeError("state must be a RecoveryState.")
    signature = _validate_signature(contract_signature)
    root = Path(recovery_root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / _checkpoint_name(state)
    if final.exists():
        raise FileExistsError(final)
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.tmp-", dir=root))
    try:
        bundle.save_policy(temporary)
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            policy_file = temporary / name
            if not policy_file.is_file() or policy_file.stat().st_size <= 0:
                raise RuntimeError(f"Checkpoint policy is missing {name}.")
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": capture_rng_state(),
                "pending_buffer_payload": (
                    None if state.pending_buffer is None else state.pending_buffer.payload
                ),
            },
            temporary / "training_state.pt",
        )
        _fsync_file(temporary / "training_state.pt")
        progress = _progress_value(state)
        _write_json(temporary / "progress.json", progress)
        _write_json(temporary / "contract_signature.json", signature)
        for path in temporary.rglob("*"):
            if path.is_file():
                _fsync_file(path)
        files = snapshot_directory(temporary)
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "progress": progress,
            "contract_signature": signature,
            "files": files,
        }
        _write_json(temporary / _MANIFEST, manifest)
        manifest_sha = sha256_file(temporary / _MANIFEST)
        (temporary / _MANIFEST_SHA).write_text(manifest_sha + "\n", encoding="ascii")
        _fsync_file(temporary / _MANIFEST_SHA)
        _fsync_directory(temporary)
        os.replace(temporary, final)
        _fsync_directory(root)
        _atomic_json(
            root / "latest.json",
            {
                "checkpoint": final.name,
                "next_source_block": state.next_source_block,
                "optimization_window_index": state.optimization_window_index,
                "completed_policy_steps": state.completed_policy_steps,
                "total_optimizer_steps": state.total_optimizer_steps,
                "final_auxiliary_flush_steps": state.final_auxiliary_flush_steps,
                "manifest_sha256": manifest_sha,
            },
        )
        return final
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Recovery {label} is corrupt: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"Recovery {label} must be an object.")
    return value


def _load_checkpoint(
    checkpoint: Path, contract_signature: Mapping[str, Any]
) -> tuple[RecoveryState, dict[str, Any]]:
    signature = _validate_signature(contract_signature)
    manifest = _read_json(checkpoint / _MANIFEST, "manifest")
    try:
        recorded_sha = (checkpoint / _MANIFEST_SHA).read_text(
            encoding="ascii"
        ).strip()
    except OSError as exc:
        raise RuntimeError("Recovery manifest digest is missing or unreadable.") from exc
    if recorded_sha != sha256_file(checkpoint / _MANIFEST):
        raise RuntimeError("Recovery manifest SHA256 mismatch.")
    if set(manifest) != {
        "schema_version",
        "progress",
        "contract_signature",
        "files",
    } or manifest["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Recovery manifest has an invalid schema.")
    if manifest["contract_signature"] != signature:
        raise RuntimeError("Recovery checkpoint belongs to a different runtime contract.")
    if _read_json(
        checkpoint / "contract_signature.json", "contract signature"
    ) != signature:
        raise RuntimeError("Recovery contract signature file differs from the manifest.")
    verify_directory_snapshot(
        checkpoint, manifest["files"], exclude=(_MANIFEST, _MANIFEST_SHA)
    )
    progress = _read_json(checkpoint / "progress.json", "progress")
    if progress != manifest["progress"]:
        raise RuntimeError("Recovery progress file differs from the manifest.")
    try:
        raw_training = torch.load(
            checkpoint / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise RuntimeError("Recovery training state is unreadable.") from exc
    if not isinstance(raw_training, dict) or set(raw_training) != {
        "optimizer",
        "scheduler",
        "rng",
        "pending_buffer_payload",
    }:
        raise ValueError("Recovery training state has an invalid schema.")
    state = _state_from_values(progress, raw_training["pending_buffer_payload"])
    if _parse_checkpoint(checkpoint) != (
        state.next_source_block,
        state.optimization_window_index,
        state.total_optimizer_steps,
    ):
        raise RuntimeError("Recovery directory name differs from saved progress.")
    training_state = {
        "optimizer": raw_training["optimizer"],
        "scheduler": raw_training["scheduler"],
        "rng": raw_training["rng"],
    }
    return state, training_state


def load_latest_recovery(
    recovery_root: Path, contract_signature: Mapping[str, Any]
) -> tuple[Path, RecoveryState, dict[str, Any]] | None:
    """Load the newest committed checkpoint and reject it if integrity fails."""

    _validate_signature(contract_signature)
    root = Path(recovery_root)
    if not root.exists():
        return None
    candidates = {
        parsed: path
        for path in root.iterdir()
        if (parsed := _parse_checkpoint(path)) is not None
    }
    if not candidates:
        return None
    checkpoint = candidates[max(candidates)]
    state, training_state = _load_checkpoint(checkpoint, contract_signature)
    return checkpoint, state, training_state


def validate_recovery_checkpoint(
    checkpoint: Path, contract_signature: Mapping[str, Any]
) -> tuple[RecoveryState, dict[str, Any]]:
    """Validate one named checkpoint without selecting or mutating another."""

    return _load_checkpoint(Path(checkpoint), contract_signature)


def restore_training_state(
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    training_state: Mapping[str, Any],
) -> None:
    if set(training_state) != {"optimizer", "scheduler", "rng"}:
        raise ValueError("Loaded recovery training state has an invalid schema.")
    optimizer.load_state_dict(training_state["optimizer"])
    scheduler.load_state_dict(training_state["scheduler"])
    restore_rng_state(training_state["rng"])


__all__ = [
    "CHECKPOINT_SCHEMA_VERSION",
    "PendingBuffer",
    "RecoveryState",
    "capture_rng_state",
    "load_latest_recovery",
    "restore_rng_state",
    "restore_training_state",
    "save_recovery_checkpoint",
    "validate_recovery_checkpoint",
]
