"""Require four successful one-step GPU gates before the full ORPO run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ksllm4rec_sft.manifest import atomic_write_json

from .contract import GATE_CUTOFFS, GATE_DATASET_NAME
from .fingerprint import implementation_fingerprint


def verify_gpu_gates(
    log_root: Path,
    report_path: Path,
    project_root: Path,
    max_reserved_gib: float = 20.0,
) -> dict[str, Any]:
    current_fingerprint = implementation_fingerprint(project_root)
    candidates: dict[str, list[tuple[str, Path, dict[str, Any]]]] = {
        name: [] for name in GATE_CUTOFFS
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
    for stage, expected_cutoff in GATE_CUTOFFS.items():
        valid: list[tuple[str, Path, dict[str, Any]]] = []
        for _, path, manifest in candidates[stage]:
            resolved = manifest.get("resolved_config", {})
            result = manifest.get("result", {})
            custom = manifest.get("custom_orpo", {})
            if manifest.get("status") != "passed":
                continue
            if int(resolved.get("cutoff_len", -1)) != expected_cutoff:
                continue
            if resolved.get("dataset") != GATE_DATASET_NAME:
                continue
            if int(resolved.get("max_steps", -1)) != 1:
                continue
            if int(custom.get("chunk_size", -1)) != 512:
                continue
            if int(result.get("optimizer_steps", 0)) != 1:
                continue
            if result.get("reference_model_used") is not False:
                continue
            if int(result.get("max_sequence_length", 0)) != expected_cutoff:
                continue
            if manifest.get("implementation_fingerprint", {}).get(
                "sha256"
            ) != current_fingerprint["sha256"]:
                continue
            if (
                float(manifest.get("peak_memory_reserved_gib", float("inf")))
                > max_reserved_gib
            ):
                continue
            valid.append((str(manifest.get("updated_at", "")), path, manifest))
        if not valid:
            failures.append(
                f"{stage}: require passed cutoff={expected_cutoff}, exact observed "
                "length, one optimizer step, reference-free ORPO, current fingerprint, "
                f"and peak reserved <= {max_reserved_gib} GiB"
            )
            continue
        _, path, manifest = max(valid, key=lambda item: item[0])
        accepted[stage] = {
            "manifest": str(path.resolve()),
            "cutoff_len": expected_cutoff,
            "optimizer_steps": manifest["result"]["optimizer_steps"],
            "max_sequence_length": manifest["result"]["max_sequence_length"],
            "peak_memory_reserved_gib": manifest["peak_memory_reserved_gib"],
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
