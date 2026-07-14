"""Validate staged GPU gates before the full-epoch run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .fingerprint import implementation_fingerprint
from .manifest import atomic_write_json


REQUIRED_GATES = {
    "gate_00512": 512,
    "gate_02048": 2048,
    "gate_08192": 8192,
    "gate_16384": 16384,
}


def verify_gpu_gates(
    log_root: Path,
    report_path: Path,
    project_root: Path,
    max_reserved_gib: float = 20.0,
) -> dict[str, Any]:
    current_fingerprint = implementation_fingerprint(project_root)
    candidates: dict[str, list[tuple[str, Path, dict[str, Any]]]] = {
        name: [] for name in REQUIRED_GATES
    }
    for path in log_root.glob("*/manifest.json"):
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        stage = manifest.get("stage")
        if stage in candidates:
            candidates[stage].append(
                (str(manifest.get("updated_at", "")), path, manifest)
            )

    accepted: dict[str, Any] = {}
    failures: list[str] = []
    for stage, expected_cutoff in REQUIRED_GATES.items():
        valid = []
        for _, path, manifest in candidates[stage]:
            resolved = manifest.get("resolved_config", {})
            result = manifest.get("result", {})
            if manifest.get("status") != "passed":
                continue
            if int(resolved.get("cutoff_len", -1)) != expected_cutoff:
                continue
            if int(resolved.get("max_steps", -1)) != 1:
                continue
            custom = manifest.get("custom_loss", {})
            if (
                float(custom.get("gamma", -1.0)) != 2.0
                or float(custom.get("item_weight", -1.0)) != 3.0
                or int(custom.get("chunk_size", -1)) != 512
            ):
                continue
            if int(result.get("optimizer_steps", 0)) < 1:
                continue
            if int(result.get("fallback_count", -1)) != 0:
                continue
            manifest_fingerprint = manifest.get("implementation_fingerprint", {})
            if manifest_fingerprint.get("sha256") != current_fingerprint["sha256"]:
                continue
            observed_length = int(result.get("max_sequence_length", 0))
            if observed_length < int(expected_cutoff * 0.95):
                continue
            if (
                float(manifest.get("peak_memory_reserved_gib", float("inf")))
                > max_reserved_gib
            ):
                continue
            valid.append((str(manifest.get("updated_at", "")), path, manifest))
        if not valid:
            failures.append(
                f"{stage}: need passed cutoff={expected_cutoff}, optimizer_steps>=1, "
                f"max_steps=1, gamma=2, item_weight=3, chunk_size=512, fallback_count=0, "
                f"observed_length>=95%, current implementation fingerprint, "
                f"peak_reserved<={max_reserved_gib} GiB"
            )
            continue
        _, path, manifest = max(valid, key=lambda item: item[0])
        accepted[stage] = {
            "manifest": str(path.resolve()),
            "cutoff_len": expected_cutoff,
            "optimizer_steps": manifest["result"]["optimizer_steps"],
            "peak_memory_reserved_gib": manifest["peak_memory_reserved_gib"],
            "max_sequence_length": manifest["result"]["max_sequence_length"],
        }

    report = {
        "status": "failed" if failures else "passed",
        "max_reserved_gib": max_reserved_gib,
        "implementation_fingerprint": current_fingerprint,
        "accepted": accepted,
        "failures": failures,
    }
    atomic_write_json(report_path, report)
    if failures:
        raise RuntimeError("GPU gates are incomplete: " + "; ".join(failures))
    return report
