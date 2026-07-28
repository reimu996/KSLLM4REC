"""Atomic, content-verified recovery checkpoints for RLOO Spec V2.0."""

from __future__ import annotations

import json
import math
import os
import random
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .rloo_integrity import (
    canonical_sha256,
    sha256_file,
    snapshot_directory,
    verify_directory_snapshot,
)


CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_INTERVAL = 100
_MANIFEST = "manifest.json"
_MANIFEST_SHA = "manifest.sha256"
_CHECKPOINT_RE = re.compile(r"checkpoint-window-(\d+)-update-(\d+)")


@dataclass(frozen=True)
class RecoveryCursor:
    """The exact next source group and optimizer state needed for continuation."""

    epoch_index: int
    next_group_offset: int
    window_step: int
    optimizer_update_step: int
    groups_completed: int
    rollout_chunk: int
    loss_chunk: int
    lambda0: float

    def __post_init__(self) -> None:
        integer_fields = (
            "epoch_index",
            "next_group_offset",
            "window_step",
            "optimizer_update_step",
            "groups_completed",
            "rollout_chunk",
            "loss_chunk",
        )
        for name in integer_fields:
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"RecoveryCursor.{name} must be an integer.")
        for name in (
            "epoch_index",
            "next_group_offset",
            "window_step",
            "optimizer_update_step",
            "groups_completed",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"RecoveryCursor.{name} must be non-negative.")
        if self.rollout_chunk <= 0 or self.loss_chunk <= 0:
            raise ValueError("Recovery chunks must be positive.")
        if self.optimizer_update_step > self.window_step:
            raise ValueError("optimizer_update_step cannot exceed window_step.")
        if self.groups_completed < self.next_group_offset:
            raise ValueError("groups_completed cannot be below next_group_offset.")
        if not isinstance(self.lambda0, (int, float)) or isinstance(self.lambda0, bool):
            raise TypeError("RecoveryCursor.lambda0 must be a number.")
        if not math.isfinite(float(self.lambda0)) or not 0.0 <= float(self.lambda0) <= 0.05:
            raise ValueError("RecoveryCursor.lambda0 must be finite and in [0, 0.05].")


def capture_rng_state() -> dict[str, Any]:
    """Capture every RNG stream used by CPU/CUDA rollout and training."""

    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise ValueError("Recovery RNG state has an invalid schema.")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        cuda_states = state["torch_cuda"]
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("Recovery CUDA RNG state has a different device count.")
        torch.cuda.set_rng_state_all(cuda_states)


def recovery_checkpoint_due(
    cursor: RecoveryCursor,
    *,
    optimizer_stepped: bool,
    epoch_finished: bool,
    interval: int = CHECKPOINT_INTERVAL,
) -> bool:
    """Return whether the just-finished event requires a recovery checkpoint."""

    if interval <= 0:
        raise ValueError("Checkpoint interval must be positive.")
    return epoch_finished or (
        optimizer_stepped
        and cursor.optimizer_update_step > 0
        and cursor.optimizer_update_step % interval == 0
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
        _fsync_directory(path.parent)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_files(root: Path) -> None:
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        with path.open("rb") as handle:
            os.fsync(handle.fileno())
    for path in sorted(
        (item for item in root.rglob("*") if item.is_dir()),
        key=lambda item: len(item.parts),
        reverse=True,
    ):
        _fsync_directory(path)
    _fsync_directory(root)


def _signature_is_self_consistent(signature: dict[str, Any]) -> bool:
    return (
        set(signature) == {"sha256", "inputs"}
        and isinstance(signature["sha256"], str)
        and len(signature["sha256"]) == 64
        and signature["sha256"] == canonical_sha256(signature["inputs"])
    )


def _validate_signature(signature: dict[str, Any]) -> None:
    if not isinstance(signature, dict) or not _signature_is_self_consistent(signature):
        raise ValueError("Runtime contract signature is invalid or self-inconsistent.")


def _lambda0_values(value: Any) -> list[float]:
    found: list[float] = []
    if isinstance(value, dict):
        for key, child in value.items():
            if key == "lambda0":
                if isinstance(child, bool) or not isinstance(child, (int, float)):
                    raise TypeError("Resolved contract lambda0 must be numeric.")
                found.append(float(child))
            found.extend(_lambda0_values(child))
    elif isinstance(value, list):
        for child in value:
            found.extend(_lambda0_values(child))
    return found


def _validate_cursor_contract(cursor: RecoveryCursor, resolved_contract: dict[str, Any]) -> None:
    if not isinstance(resolved_contract, dict):
        raise TypeError("Resolved contract must be a mapping.")
    values = _lambda0_values(resolved_contract)
    if not values:
        raise ValueError("Resolved contract must contain the frozen lambda0.")
    if any(not math.isclose(value, float(cursor.lambda0), rel_tol=0.0, abs_tol=1e-15) for value in values):
        raise RuntimeError("Recovery cursor lambda0 differs from the resolved contract.")


def _checkpoint_name(cursor: RecoveryCursor) -> str:
    return (
        f"checkpoint-window-{cursor.window_step:06d}"
        f"-update-{cursor.optimizer_update_step:06d}"
    )


def save_policy_atomic(bundle: Any, output_dir: Path) -> tuple[Path, Path]:
    """Atomically publish one epoch adapter directory."""

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        bundle.save_policy(temporary)
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            path = temporary / name
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Temporary policy adapter is missing {name}.")
        _fsync_files(temporary)
        os.replace(temporary, output_dir)
        _fsync_directory(output_dir.parent)
        return (
            output_dir / "adapter_model.safetensors",
            output_dir / "adapter_config.json",
        )
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def save_recovery_checkpoint(
    bundle: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    cursor: RecoveryCursor,
    recovery_root: Path,
    contract_signature: dict[str, Any],
    resolved_contract: dict[str, Any],
) -> Path:
    """Atomically save all state required to reproduce the next rollout."""

    _validate_signature(contract_signature)
    _validate_cursor_contract(cursor, resolved_contract)
    recovery_root = Path(recovery_root)
    recovery_root.mkdir(parents=True, exist_ok=True)
    final = recovery_root / _checkpoint_name(cursor)
    if final.exists():
        raise FileExistsError(final)
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.tmp-", dir=recovery_root))
    try:
        bundle.save_policy(temporary)
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            path = temporary / name
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Recovery checkpoint is missing {name}.")

        training_path = temporary / "training_state.pt"
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "rng": capture_rng_state(),
            },
            training_path,
        )
        _write_json(temporary / "cursor.json", asdict(cursor))
        _write_json(temporary / "resolved_contract.json", resolved_contract)
        _write_json(temporary / "contract_signature.json", contract_signature)

        payload_files = snapshot_directory(temporary)
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "cursor": asdict(cursor),
            "resolved_contract_sha256": canonical_sha256(resolved_contract),
            "contract_signature": contract_signature,
            "files": payload_files,
        }
        _write_json(temporary / _MANIFEST, manifest)
        manifest_sha = sha256_file(temporary / _MANIFEST)
        with (temporary / _MANIFEST_SHA).open("w", encoding="ascii") as handle:
            handle.write(manifest_sha + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        _fsync_files(temporary)
        os.replace(temporary, final)
        _fsync_directory(recovery_root)
        _atomic_json(
            recovery_root / "latest.json",
            {
                "checkpoint": final.name,
                "window_step": cursor.window_step,
                "optimizer_update_step": cursor.optimizer_update_step,
                "manifest_sha256": manifest_sha,
            },
        )
        return final
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _parse_checkpoint(path: Path) -> tuple[int, int] | None:
    match = _CHECKPOINT_RE.fullmatch(path.name)
    if not path.is_dir() or match is None:
        return None
    return int(match.group(1)), int(match.group(2))


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Recovery {label} is corrupt: {path}") from exc


def _validate_checkpoint(
    checkpoint: Path,
    expected_contract_signature: dict[str, Any],
    expected_resolved_contract: dict[str, Any],
    *,
    pointer_manifest_sha: str | None,
) -> tuple[RecoveryCursor, dict[str, Any]]:
    manifest_path = checkpoint / _MANIFEST
    digest_path = checkpoint / _MANIFEST_SHA
    if not manifest_path.is_file() or not digest_path.is_file():
        raise RuntimeError(f"Recovery checkpoint is incomplete: {checkpoint}")
    try:
        recorded_manifest_sha = digest_path.read_text(encoding="ascii").strip()
    except OSError as exc:
        raise RuntimeError(f"Recovery manifest digest is corrupt: {digest_path}") from exc
    actual_manifest_sha = sha256_file(manifest_path)
    if (
        len(recorded_manifest_sha) != 64
        or recorded_manifest_sha != actual_manifest_sha
        or (
            pointer_manifest_sha is not None
            and pointer_manifest_sha != actual_manifest_sha
        )
    ):
        raise RuntimeError("Recovery manifest SHA256 mismatch.")
    manifest = _read_json(manifest_path, "manifest")
    if not isinstance(manifest, dict) or set(manifest) != {
        "schema_version",
        "cursor",
        "resolved_contract_sha256",
        "contract_signature",
        "files",
    }:
        raise ValueError("Recovery manifest has an invalid schema.")
    if manifest["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Unsupported recovery manifest schema.")
    if manifest["contract_signature"] != expected_contract_signature:
        raise RuntimeError("Recovery checkpoint belongs to a different runtime contract.")
    signature_file = _read_json(
        checkpoint / "contract_signature.json", "contract signature"
    )
    if signature_file != expected_contract_signature:
        raise RuntimeError("Recovery contract signature file does not match.")
    verify_directory_snapshot(
        checkpoint,
        manifest["files"],
        exclude=(_MANIFEST, _MANIFEST_SHA),
    )

    cursor_file = _read_json(checkpoint / "cursor.json", "cursor")
    if cursor_file != manifest["cursor"]:
        raise RuntimeError("Recovery cursor file and manifest differ.")
    cursor = RecoveryCursor(**cursor_file)
    parsed = _parse_checkpoint(checkpoint)
    if parsed != (cursor.window_step, cursor.optimizer_update_step):
        raise RuntimeError("Recovery directory name and cursor steps differ.")
    resolved_contract = _read_json(
        checkpoint / "resolved_contract.json", "resolved contract"
    )
    if canonical_sha256(resolved_contract) != manifest["resolved_contract_sha256"]:
        raise RuntimeError("Recovery resolved contract SHA256 differs from manifest.")
    if canonical_sha256(resolved_contract) != canonical_sha256(
        expected_resolved_contract
    ):
        raise RuntimeError(
            "Recovery checkpoint belongs to a different resolved contract."
        )
    _validate_cursor_contract(cursor, resolved_contract)

    try:
        training = torch.load(
            checkpoint / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise RuntimeError("Recovery training state is unreadable.") from exc
    if not isinstance(training, dict) or set(training) != {
        "optimizer",
        "scheduler",
        "rng",
    }:
        raise ValueError("Recovery training state has an invalid schema.")
    training["resolved_contract"] = resolved_contract
    training["contract_signature"] = signature_file
    return cursor, training


def load_latest_recovery(
    recovery_root: Path,
    expected_contract_signature: dict[str, Any],
    expected_resolved_contract: dict[str, Any],
) -> tuple[Path, RecoveryCursor, dict[str, Any]] | None:
    """Load the newest complete checkpoint; never fall back from corruption."""

    _validate_signature(expected_contract_signature)
    if not isinstance(expected_resolved_contract, dict):
        raise TypeError("Expected resolved contract must be a mapping.")
    recovery_root = Path(recovery_root)
    latest_path = recovery_root / "latest.json"
    latest: dict[str, Any] | None = None
    if latest_path.exists():
        latest = _read_json(latest_path, "latest pointer")
        if not isinstance(latest, dict) or set(latest) != {
            "checkpoint",
            "window_step",
            "optimizer_update_step",
            "manifest_sha256",
        }:
            raise ValueError("Recovery latest pointer has an invalid schema.")

    candidates: dict[tuple[int, int], Path] = {}
    if recovery_root.exists():
        for path in recovery_root.iterdir():
            parsed = _parse_checkpoint(path)
            if parsed is not None:
                candidates[parsed] = path
    if latest is None and not candidates:
        return None

    pointed_key: tuple[int, int] | None = None
    if latest is not None:
        pointed_key = (
            int(latest["window_step"]),
            int(latest["optimizer_update_step"]),
        )
        expected_name = (
            f"checkpoint-window-{pointed_key[0]:06d}"
            f"-update-{pointed_key[1]:06d}"
        )
        if latest["checkpoint"] != expected_name:
            raise RuntimeError("Recovery pointer name and steps differ.")

    newest_key = max(candidates) if candidates else None
    if pointed_key is None or (newest_key is not None and newest_key > pointed_key):
        selected_key = newest_key
        assert selected_key is not None
        checkpoint = candidates[selected_key]
        pointer_manifest_sha = None
        promote = True
    else:
        selected_key = pointed_key
        assert selected_key is not None and latest is not None
        checkpoint = recovery_root / str(latest["checkpoint"])
        pointer_manifest_sha = str(latest["manifest_sha256"])
        promote = False

    cursor, training = _validate_checkpoint(
        checkpoint,
        expected_contract_signature,
        expected_resolved_contract,
        pointer_manifest_sha=pointer_manifest_sha,
    )
    if selected_key != (cursor.window_step, cursor.optimizer_update_step):
        raise RuntimeError("Recovery selection and cursor steps differ.")
    if promote:
        _atomic_json(
            latest_path,
            {
                "checkpoint": checkpoint.name,
                "window_step": cursor.window_step,
                "optimizer_update_step": cursor.optimizer_update_step,
                "manifest_sha256": sha256_file(checkpoint / _MANIFEST),
            },
        )
    return checkpoint, cursor, training


def restore_training_state(
    optimizer: torch.optim.Optimizer,
    scheduler: Any | None,
    training_state: dict[str, Any],
) -> None:
    """Restore optimizer, optional scheduler, and RNG after model loading."""

    required = {
        "optimizer",
        "scheduler",
        "rng",
        "resolved_contract",
        "contract_signature",
    }
    if set(training_state) != required:
        raise ValueError("Loaded recovery state has an invalid schema.")
    optimizer.load_state_dict(training_state["optimizer"])
    scheduler_state = training_state["scheduler"]
    if scheduler is None:
        if scheduler_state is not None:
            raise RuntimeError("Checkpoint has scheduler state but runtime has no scheduler.")
    else:
        if scheduler_state is None:
            raise RuntimeError("Checkpoint has no scheduler state for this runtime.")
        scheduler.load_state_dict(scheduler_state)
    restore_rng_state(training_state["rng"])
