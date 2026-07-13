"""Strict checkpoint discovery for resumable full-epoch training."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_CHECKPOINT_RE = re.compile(r"^checkpoint-(\d+)$")
REQUIRED_CHECKPOINT_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "optimizer.pt",
    "scheduler.pt",
    "trainer_state.json",
    "rng_state.pth",
)


def validate_checkpoint(checkpoint: Path) -> dict[str, Any]:
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
    return {
        "path": str(checkpoint),
        "step": step,
        "files": {
            name: (checkpoint / name).stat().st_size
            for name in REQUIRED_CHECKPOINT_FILES
        },
    }


def resolve_resume_checkpoint(output_dir: Path) -> dict[str, Any] | None:
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
        if entries:
            raise RuntimeError(
                f"Full-run output directory is non-empty but has no resumable checkpoint: {output_dir}"
            )
        return None
    _, latest = max(candidates, key=lambda value: value[0])
    return validate_checkpoint(latest)
