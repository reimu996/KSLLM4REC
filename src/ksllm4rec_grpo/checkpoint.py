"""Atomic recovery and epoch adapter checkpoints for GRPO V3.1."""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class RecoveryCursor:
    epoch_index: int
    next_group_offset: int
    global_step: int
    groups_completed: int
    rollout_chunk: int
    loss_chunk: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all()
        if torch.cuda.is_available()
        else [],
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    if set(state) != required:
        raise ValueError("Recovery RNG state has an invalid schema.")
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch_cpu"])
    if torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def save_policy_atomic(bundle: Any, output_dir: Path) -> tuple[Path, Path]:
    """Save one policy adapter through a validated temporary directory."""

    output_dir = Path(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(output_dir)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        weights, config = bundle.save_policy(temporary)
        if not weights.is_file() or not config.is_file():
            raise RuntimeError("Temporary policy adapter is incomplete.")
        os.rename(temporary, output_dir)
        return output_dir / weights.name, output_dir / config.name
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def save_recovery_checkpoint(
    bundle: Any,
    optimizer: torch.optim.Optimizer,
    scheduler: Any,
    cursor: RecoveryCursor,
    recovery_root: Path,
    contract_signature: dict[str, Any],
) -> Path:
    """Atomically save a complete checkpoint and move the latest pointer."""

    recovery_root = Path(recovery_root)
    recovery_root.mkdir(parents=True, exist_ok=True)
    final = recovery_root / f"checkpoint-step-{cursor.global_step:06d}"
    if final.exists():
        raise FileExistsError(final)
    temporary = Path(tempfile.mkdtemp(prefix=f".{final.name}.tmp-", dir=recovery_root))
    try:
        bundle.save_policy(temporary)
        training_path = temporary / "training_state.pt"
        torch.save(
            {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "rng": capture_rng_state(),
            },
            training_path,
        )
        state_path = temporary / "cursor.json"
        state_path.write_text(
            json.dumps(asdict(cursor), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        files = {}
        for name in (
            "adapter_model.safetensors",
            "adapter_config.json",
            "training_state.pt",
            "cursor.json",
        ):
            path = temporary / name
            if not path.is_file() or path.stat().st_size == 0:
                raise RuntimeError(f"Recovery checkpoint missing {name}.")
            files[name] = {
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            }
        (temporary / "manifest.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "contract_signature": contract_signature,
                    "cursor": asdict(cursor),
                    "files": files,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        os.rename(temporary, final)
        _atomic_json(
            recovery_root / "latest.json",
            {"checkpoint": final.name, "global_step": cursor.global_step},
        )
        return final
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_latest_recovery(
    recovery_root: Path,
    expected_contract_signature: dict[str, Any],
) -> tuple[Path, RecoveryCursor, dict[str, Any]] | None:
    """Load exactly the pointed checkpoint; never fall back to an older one."""

    recovery_root = Path(recovery_root)
    latest_path = recovery_root / "latest.json"
    latest = None
    if latest_path.exists():
        latest = json.loads(latest_path.read_text(encoding="utf-8"))
        if set(latest) != {"checkpoint", "global_step"}:
            raise ValueError("Recovery latest pointer has an invalid schema.")
    candidates: dict[int, Path] = {}
    if recovery_root.exists():
        for path in recovery_root.glob("checkpoint-step-*"):
            suffix = path.name.removeprefix("checkpoint-step-")
            if path.is_dir() and suffix.isdigit():
                candidates[int(suffix)] = path
    if latest is None and not candidates:
        return None
    pointed_step = int(latest["global_step"]) if latest is not None else -1
    selected_step = max([pointed_step, *candidates])
    if selected_step > pointed_step:
        checkpoint = candidates[selected_step]
        selected = {"checkpoint": checkpoint.name, "global_step": selected_step}
        promote = True
    else:
        checkpoint = recovery_root / latest["checkpoint"]
        selected = latest
        promote = False
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    if set(manifest) != {
        "schema_version",
        "contract_signature",
        "cursor",
        "files",
    }:
        raise ValueError("Recovery manifest has an invalid schema.")
    if manifest["schema_version"] != 2:
        raise ValueError("Unsupported recovery manifest schema.")
    if manifest["contract_signature"] != expected_contract_signature:
        raise RuntimeError(
            "Recovery checkpoint belongs to a different runtime contract."
        )
    for name, record in manifest["files"].items():
        path = checkpoint / name
        if (
            not path.is_file()
            or path.stat().st_size != record["size"]
            or _sha256(path) != record["sha256"]
        ):
            raise RuntimeError(f"Latest recovery checkpoint is corrupt: {name}.")
    cursor = RecoveryCursor(**manifest["cursor"])
    cursor_file = json.loads((checkpoint / "cursor.json").read_text(encoding="utf-8"))
    if cursor_file != manifest["cursor"]:
        raise RuntimeError("Recovery cursor file and manifest differ.")
    if cursor.global_step != selected["global_step"]:
        raise RuntimeError("Recovery pointer and cursor global steps differ.")
    training = torch.load(
        checkpoint / "training_state.pt", map_location="cpu", weights_only=False
    )
    if set(training) != {"optimizer", "scheduler", "rng"}:
        raise ValueError("Recovery training state has an invalid schema.")
    if promote:
        _atomic_json(latest_path, selected)
    return checkpoint, cursor, training
