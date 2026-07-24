"""Atomic recovery checkpoints committed only at complete window boundaries."""

from __future__ import annotations

import json
import os
import random
import re
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from ksllm4rec_rloo.integrity import (
    canonical_sha256,
    sha256_file,
    snapshot_directory,
    verify_directory_snapshot,
)

from . import contract
from .sampling import SourceCursor


CHECKPOINT_SCHEMA_VERSION = 1
_MANIFEST = "manifest.json"
_MANIFEST_SHA = "manifest.sha256"
_CHECKPOINT_RE = re.compile(r"checkpoint-window-(\d+)-update-(\d+)")


@dataclass(frozen=True)
class RecoveryCursor:
    next_window_index: int
    optimizer_update_step: int
    source_cursor: SourceCursor
    window_log_rows: int
    group_log_rows: int

    def __post_init__(self) -> None:
        for name in (
            "next_window_index",
            "optimizer_update_step",
            "window_log_rows",
            "group_log_rows",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"RecoveryCursor.{name} must be a non-negative integer.")
        if not isinstance(self.source_cursor, SourceCursor):
            raise TypeError("source_cursor must be a SourceCursor.")
        if self.optimizer_update_step != (
            self.next_window_index * contract.OPTIMIZER_UPDATES_PER_WINDOW
        ):
            raise ValueError("Optimizer updates must equal four per complete window.")
        if self.window_log_rows != self.next_window_index:
            raise ValueError("Window log rows must equal completed windows.")
        if self.group_log_rows != (
            self.next_window_index * contract.EFFECTIVE_GROUPS_PER_WINDOW
        ):
            raise ValueError("Group log rows must equal 32 per complete window.")


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


def recovery_checkpoint_due(
    cursor: RecoveryCursor,
    *,
    epoch_finished: bool,
    interval_updates: int = 100,
) -> bool:
    if interval_updates <= 0:
        raise ValueError("interval_updates must be positive.")
    return bool(epoch_finished) or (
        cursor.optimizer_update_step > 0
        and cursor.optimizer_update_step % int(interval_updates) == 0
    )


def _validate_signature(signature: Mapping[str, Any]) -> None:
    if (
        set(signature) != {"sha256", "inputs"}
        or signature["sha256"] != canonical_sha256(signature["inputs"])
    ):
        raise ValueError("Runtime contract signature is invalid.")


def _write_json(path: Path, value: Any) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _cursor_value(cursor: RecoveryCursor) -> dict[str, Any]:
    return asdict(cursor)


def _cursor_from_value(value: Mapping[str, Any]) -> RecoveryCursor:
    if set(value) != {
        "next_window_index",
        "optimizer_update_step",
        "source_cursor",
        "window_log_rows",
        "group_log_rows",
    }:
        raise ValueError("Recovery cursor has an invalid schema.")
    source = value["source_cursor"]
    if not isinstance(source, Mapping) or set(source) != {"cycle_index", "offset"}:
        raise ValueError("Recovery source cursor has an invalid schema.")
    return RecoveryCursor(
        next_window_index=value["next_window_index"],
        optimizer_update_step=value["optimizer_update_step"],
        source_cursor=SourceCursor(**source),
        window_log_rows=value["window_log_rows"],
        group_log_rows=value["group_log_rows"],
    )


def _checkpoint_name(cursor: RecoveryCursor) -> str:
    return (
        f"checkpoint-window-{cursor.next_window_index:06d}"
        f"-update-{cursor.optimizer_update_step:06d}"
    )


def save_recovery_checkpoint(
    bundle: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    cursor: RecoveryCursor,
    recovery_root: Path,
    contract_signature: Mapping[str, Any],
) -> Path:
    _validate_signature(contract_signature)
    root = Path(recovery_root)
    root.mkdir(parents=True, exist_ok=True)
    final = root / _checkpoint_name(cursor)
    if final.exists():
        raise FileExistsError(final)
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.tmp-", dir=root))
    try:
        bundle.save_policy(temporary)
        for name in ("adapter_model.safetensors", "adapter_config.json"):
            path = temporary / name
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Checkpoint policy is missing {name}.")
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": capture_rng_state(),
            },
            temporary / "training_state.pt",
        )
        _write_json(temporary / "cursor.json", _cursor_value(cursor))
        _write_json(temporary / "contract_signature.json", dict(contract_signature))
        files = snapshot_directory(temporary)
        manifest = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "cursor": _cursor_value(cursor),
            "contract_signature": dict(contract_signature),
            "files": files,
        }
        _write_json(temporary / _MANIFEST, manifest)
        manifest_sha = sha256_file(temporary / _MANIFEST)
        (temporary / _MANIFEST_SHA).write_text(manifest_sha + "\n", encoding="ascii")
        os.replace(temporary, final)
        _atomic_json(
            root / "latest.json",
            {
                "checkpoint": final.name,
                "next_window_index": cursor.next_window_index,
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
) -> tuple[RecoveryCursor, dict[str, Any]]:
    manifest = _read_json(checkpoint / _MANIFEST, "manifest")
    recorded_sha = (checkpoint / _MANIFEST_SHA).read_text(encoding="ascii").strip()
    actual_sha = sha256_file(checkpoint / _MANIFEST)
    if recorded_sha != actual_sha:
        raise RuntimeError("Recovery manifest SHA256 mismatch.")
    if set(manifest) != {
        "schema_version",
        "cursor",
        "contract_signature",
        "files",
    } or manifest["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError("Recovery manifest has an invalid schema.")
    if manifest["contract_signature"] != dict(contract_signature):
        raise RuntimeError("Recovery checkpoint belongs to a different runtime contract.")
    signature_file = _read_json(
        checkpoint / "contract_signature.json", "contract signature"
    )
    if signature_file != dict(contract_signature):
        raise RuntimeError("Recovery signature file differs from the runtime contract.")
    verify_directory_snapshot(
        checkpoint, manifest["files"], exclude=(_MANIFEST, _MANIFEST_SHA)
    )
    cursor_file = _read_json(checkpoint / "cursor.json", "cursor")
    if cursor_file != manifest["cursor"]:
        raise RuntimeError("Recovery cursor and manifest differ.")
    cursor = _cursor_from_value(cursor_file)
    if _parse_checkpoint(checkpoint) != (
        cursor.next_window_index,
        cursor.optimizer_update_step,
    ):
        raise RuntimeError("Recovery directory name and cursor differ.")
    try:
        state = torch.load(
            checkpoint / "training_state.pt",
            map_location="cpu",
            weights_only=False,
        )
    except Exception as exc:
        raise RuntimeError("Recovery training state is unreadable.") from exc
    if not isinstance(state, dict) or set(state) != {"optimizer", "scheduler", "rng"}:
        raise ValueError("Recovery training state has an invalid schema.")
    return cursor, state


def load_latest_recovery(
    recovery_root: Path, contract_signature: Mapping[str, Any]
) -> tuple[Path, RecoveryCursor, dict[str, Any]] | None:
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
    key = max(candidates)
    checkpoint = candidates[key]
    cursor, state = _load_checkpoint(checkpoint, contract_signature)
    manifest_sha = sha256_file(checkpoint / _MANIFEST)
    _atomic_json(
        root / "latest.json",
        {
            "checkpoint": checkpoint.name,
            "next_window_index": cursor.next_window_index,
            "optimizer_update_step": cursor.optimizer_update_step,
            "manifest_sha256": manifest_sha,
        },
    )
    return checkpoint, cursor, state


def validate_recovery_checkpoint(
    checkpoint: Path, contract_signature: Mapping[str, Any]
) -> tuple[RecoveryCursor, dict[str, Any]]:
    """Validate one named complete checkpoint without mutating latest.json."""

    _validate_signature(contract_signature)
    return _load_checkpoint(Path(checkpoint), contract_signature)


def restore_training_state(
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    state: Mapping[str, Any],
) -> None:
    if set(state) != {"optimizer", "scheduler", "rng"}:
        raise ValueError("Loaded recovery state has an invalid schema.")
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    restore_rng_state(state["rng"])


__all__ = [
    "RecoveryCursor",
    "capture_rng_state",
    "load_latest_recovery",
    "recovery_checkpoint_due",
    "restore_rng_state",
    "restore_training_state",
    "save_recovery_checkpoint",
    "validate_recovery_checkpoint",
]
