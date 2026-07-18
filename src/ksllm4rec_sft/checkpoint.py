"""Strict checkpoint discovery for resumable full-epoch training."""

from __future__ import annotations

import json
import hashlib
import re
from copy import deepcopy
from pathlib import Path
from typing import Any

from transformers import TrainerCallback

from .integrity import artifact_identity

_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
REQUIRED_CHECKPOINT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state.pth",
)
CHECKPOINT_BINDING_FILE = "run_binding.json"
RUN_BINDING_FILE = ".sft_run_binding.json"
BINDING_SCHEMA_VERSION = 1


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(value: Any) -> Any:
    return json.loads(
        json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)
    )


def _checkpoint_file_snapshot(checkpoint: Path) -> dict[str, dict[str, Any]]:
    snapshot = {}
    for name in REQUIRED_CHECKPOINT_FILES:
        path = checkpoint / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(
                f"Checkpoint {checkpoint} is incomplete; missing or empty: {name}"
            )
        snapshot[name] = {"size": path.stat().st_size, "sha256": _sha256_file(path)}
    return snapshot


def build_run_identity(
    *,
    profile,
    output_dir: Path,
    input_integrity: dict[str, Any],
    resolved_config: dict[str, Any],
    custom_loss: dict[str, Any],
    implementation_fingerprint: dict[str, Any],
) -> dict[str, Any]:
    config = deepcopy(resolved_config)
    config.pop("resume_from_checkpoint", None)
    return {
        "schema_version": BINDING_SCHEMA_VERSION,
        "profile": profile.name,
        "stage": profile.full_stage,
        "output_dir": str(output_dir.resolve()),
        "input_artifact_identity": _canonical(artifact_identity(input_integrity)),
        "resolved_config": _canonical(config),
        "custom_loss": _canonical(custom_loss),
        "implementation_fingerprint": _canonical(implementation_fingerprint),
    }


def _atomic_binding(path: Path, value: dict[str, Any]) -> None:
    from .manifest import atomic_write_json

    atomic_write_json(path, value)


def write_run_binding(output_dir: Path, run_identity: dict[str, Any]) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    binding = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "run_identity": _canonical(run_identity),
    }
    _atomic_binding(output_dir / RUN_BINDING_FILE, binding)
    return binding


def validate_run_binding(
    output_dir: Path, *, expected_run_identity: dict[str, Any]
) -> dict[str, Any]:
    path = output_dir.resolve() / RUN_BINDING_FILE
    if not path.is_file() or path.stat().st_size == 0:
        raise RuntimeError(
            f"Full-run output has no {RUN_BINDING_FILE}; refusing resume."
        )
    try:
        binding = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Full-run binding is invalid: {path}") from exc
    if binding.get("schema_version") != BINDING_SCHEMA_VERSION:
        raise RuntimeError("Unsupported full-run binding schema.")
    if _canonical(binding.get("run_identity", {})) != _canonical(expected_run_identity):
        raise RuntimeError("Full-run binding does not match the current run.")
    return binding


def write_checkpoint_binding(
    checkpoint: Path, *, run_identity: dict[str, Any], origin_manifest: Path
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    snapshot = _checkpoint_file_snapshot(checkpoint)
    binding = {
        "schema_version": BINDING_SCHEMA_VERSION,
        "run_identity": _canonical(run_identity),
        "origin_manifest": str(origin_manifest.resolve()),
        "checkpoint_step": int(
            re.fullmatch(r"checkpoint-(\d+)", checkpoint.name).group(1)
        ),
        "files": snapshot,
    }
    _atomic_binding(checkpoint / CHECKPOINT_BINDING_FILE, binding)
    return binding


def _validate_origin_manifest(
    binding: dict[str, Any], *, expected_run_identity: dict[str, Any], log_root: Path
) -> None:
    origin = Path(str(binding.get("origin_manifest", ""))).resolve()
    log_root = log_root.resolve()
    if not origin.is_relative_to(log_root) or origin.name != "manifest.json":
        raise RuntimeError(
            f"Checkpoint binding origin manifest is outside the run log root: {origin}"
        )
    try:
        manifest = json.loads(origin.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Checkpoint origin manifest is unreadable: {origin}"
        ) from exc
    if manifest.get("status") not in {"running", "failed", "passed"}:
        raise RuntimeError(
            f"Checkpoint origin manifest has invalid status: {manifest.get('status')!r}"
        )
    if manifest.get("profile") != expected_run_identity["profile"]:
        raise RuntimeError(
            "Checkpoint origin manifest profile does not match run binding."
        )
    if manifest.get("stage") != expected_run_identity["stage"]:
        raise RuntimeError(
            "Checkpoint origin manifest stage does not match run binding."
        )
    resolved = deepcopy(manifest.get("resolved_config", {}))
    resolved.pop("resume_from_checkpoint", None)
    expected_config = expected_run_identity["resolved_config"]
    if _canonical(resolved) != _canonical(expected_config):
        raise RuntimeError(
            "Checkpoint origin manifest resolved config does not match run binding."
        )
    if (
        str(Path(resolved.get("output_dir", "")).resolve())
        != expected_run_identity["output_dir"]
    ):
        raise RuntimeError(
            "Checkpoint origin manifest output directory does not match run binding."
        )
    if (
        artifact_identity(manifest.get("input_integrity", {}))
        != expected_run_identity["input_artifact_identity"]
    ):
        raise RuntimeError(
            "Checkpoint origin manifest input artifact identity does not match run binding."
        )
    if _canonical(manifest.get("implementation_fingerprint", {})) != _canonical(
        expected_run_identity["implementation_fingerprint"]
    ):
        raise RuntimeError(
            "Checkpoint origin manifest implementation fingerprint does not match run binding."
        )
    if _canonical(manifest.get("run_identity", {})) != _canonical(
        expected_run_identity
    ):
        raise RuntimeError(
            "Checkpoint origin manifest run identity does not match checkpoint binding."
        )


def _validate_checkpoint_binding(
    checkpoint: Path, *, expected_run_identity: dict[str, Any], log_root: Path
) -> dict[str, Any]:
    binding_path = checkpoint / CHECKPOINT_BINDING_FILE
    if not binding_path.is_file() or binding_path.stat().st_size == 0:
        raise RuntimeError(
            f"Checkpoint {checkpoint} has no {CHECKPOINT_BINDING_FILE}; refusing resume."
        )
    try:
        binding = json.loads(binding_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Checkpoint binding is invalid: {binding_path}") from exc
    if binding.get("schema_version") != BINDING_SCHEMA_VERSION:
        raise RuntimeError("Unsupported checkpoint binding schema.")
    if _canonical(binding.get("run_identity", {})) != _canonical(expected_run_identity):
        raise RuntimeError("Checkpoint run identity does not match the current run.")
    match = _CHECKPOINT_RE.fullmatch(checkpoint.name)
    if match is None or int(binding.get("checkpoint_step", -1)) != int(match.group(1)):
        raise RuntimeError("Checkpoint binding step does not match its directory.")
    expected_files = binding.get("files", {})
    actual_files = _checkpoint_file_snapshot(checkpoint)
    if expected_files != actual_files:
        raise RuntimeError("Checkpoint file hashes do not match its binding.")
    _validate_origin_manifest(
        binding, expected_run_identity=expected_run_identity, log_root=log_root
    )
    return binding


def validate_checkpoint(
    checkpoint: Path,
    *,
    expected_run_identity: dict[str, Any] | None = None,
    log_root: Path | None = None,
) -> dict[str, Any]:
    checkpoint = checkpoint.resolve()
    match = _CHECKPOINT_RE.fullmatch(checkpoint.name)
    if not checkpoint.is_dir() or match is None:
        raise RuntimeError(f"Invalid checkpoint directory: {checkpoint}")
    step = int(match.group(1))
    missing_or_empty = [
        name
        for name in REQUIRED_CHECKPOINT_FILES
        if not (checkpoint / name).is_file() or (checkpoint / name).stat().st_size == 0
    ]
    if missing_or_empty:
        raise RuntimeError(
            f"Checkpoint {checkpoint} is incomplete; missing or empty: {missing_or_empty}"
        )
    try:
        trainer_state = json.loads(
            (checkpoint / "trainer_state.json").read_text(encoding="utf-8")
        )
        json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"Checkpoint {checkpoint} contains invalid JSON: {exc}"
        ) from exc
    state_step = int(trainer_state.get("global_step", -1))
    if state_step != step:
        raise RuntimeError(
            f"Checkpoint step mismatch: directory={step}, trainer_state={state_step}"
        )
    if expected_run_identity is not None and log_root is None:
        raise RuntimeError("log_root is required for bound checkpoint validation.")
    return {
        "path": str(checkpoint),
        "step": step,
        "files": {
            name: (checkpoint / name).stat().st_size
            for name in REQUIRED_CHECKPOINT_FILES
        },
        "binding": (
            _validate_checkpoint_binding(
                checkpoint,
                expected_run_identity=expected_run_identity,
                log_root=log_root,
            )
            if expected_run_identity is not None and log_root is not None
            else None
        ),
    }


def resolve_resume_checkpoint(
    output_dir: Path,
    *,
    expected_run_identity: dict[str, Any] | None = None,
    log_root: Path | None = None,
) -> dict[str, Any] | None:
    """Return the highest valid checkpoint; never fall back past a corrupt latest one."""

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    entries = list(output_dir.iterdir())
    candidates: list[tuple[int, Path]] = []
    for path in entries:
        match = _CHECKPOINT_RE.fullmatch(path.name)
        if match is not None and path.is_dir():
            candidates.append((int(match.group(1)), path))
    if not candidates:
        if expected_run_identity is not None and set(path.name for path in entries) == {
            RUN_BINDING_FILE
        }:
            if log_root is None:
                raise RuntimeError("log_root is required for bound checkpoint resume.")
            validate_run_binding(
                output_dir, expected_run_identity=expected_run_identity
            )
            return None
        if entries:
            raise RuntimeError(
                f"Full-run output directory is non-empty but has no resumable checkpoint: {output_dir}"
            )
        return None
    _, latest = max(candidates, key=lambda value: value[0])
    if expected_run_identity is not None and log_root is None:
        raise RuntimeError("log_root is required for bound checkpoint resume.")
    if expected_run_identity is not None:
        validate_run_binding(output_dir, expected_run_identity=expected_run_identity)
    return validate_checkpoint(
        latest,
        expected_run_identity=expected_run_identity,
        log_root=log_root,
    )


class CheckpointBindingCallback(TrainerCallback):
    """Write an input/config/hash binding after each Trainer checkpoint save."""

    def __init__(self, *, run_identity: dict[str, Any], origin_manifest: Path) -> None:
        self.run_identity = _canonical(run_identity)
        self.origin_manifest = origin_manifest.resolve()

    def on_save(self, args, state, control, **kwargs):
        if not getattr(args, "should_save", True):
            return control
        checkpoint = Path(args.output_dir) / f"checkpoint-{int(state.global_step)}"
        write_checkpoint_binding(
            checkpoint,
            run_identity=self.run_identity,
            origin_manifest=self.origin_manifest,
        )
        return control
