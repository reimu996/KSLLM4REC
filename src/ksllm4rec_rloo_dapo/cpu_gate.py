"""CPU regression gate bound to the exact runtime signature."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping


CPU_GATE_SCHEMA_VERSION = 1
CPU_SUITES = ("rloo_dapo", "rloo", "sft", "grpo", "orpo")
_COUNT_PATTERN = re.compile(r"Ran (\d+) tests? in ")


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def run_cpu_gate(
    signature: Mapping[str, Any],
    *,
    output: Path,
    log: Path,
) -> dict[str, Any]:
    """Run every approved CPU suite and publish one content-bound report."""

    root = _project_root()
    environment = os.environ.copy()
    source = str(root / "src")
    environment["PYTHONPATH"] = (
        source
        if not environment.get("PYTHONPATH")
        else f"{source}{os.pathsep}{environment['PYTHONPATH']}"
    )
    records: list[dict[str, Any]] = []
    log_parts: list[str] = []
    started = time.perf_counter()
    for suite in CPU_SUITES:
        suite_dir = root / "tests" / suite
        test_file_count = len(tuple(suite_dir.glob("test_*.py")))
        command = (
            sys.executable,
            "-m",
            "unittest",
            "discover",
            "-s",
            str(suite_dir),
            "-v",
        )
        suite_started = time.perf_counter()
        result = subprocess.run(
            command,
            cwd=root,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
        )
        elapsed = time.perf_counter() - suite_started
        matches = _COUNT_PATTERN.findall(result.stdout)
        test_count = int(matches[-1]) if len(matches) == 1 else -1
        records.append(
            {
                "suite": suite,
                "returncode": int(result.returncode),
                "test_count": test_count,
                "test_file_count": test_file_count,
                "elapsed_seconds": float(elapsed),
            }
        )
        log_parts.append(f"===== {suite} =====\n{result.stdout.rstrip()}\n")

    log_path = Path(log).expanduser().resolve()
    _atomic_write(log_path, "\n".join(log_parts) + "\n")
    passed = all(
        record["returncode"] == 0
        and record["test_count"] > 0
        and record["test_file_count"] > 0
        for record in records
    )
    report = {
        "schema_version": CPU_GATE_SCHEMA_VERSION,
        "gate": "cpu",
        "passed": passed,
        "runtime_signature_sha256": signature["sha256"],
        "suite_order": list(CPU_SUITES),
        "suites": records,
        "total_tests": sum(record["test_count"] for record in records),
        "elapsed_seconds": float(time.perf_counter() - started),
        "log_path": str(log_path),
        "log_sha256": _sha256_file(log_path),
    }
    _atomic_write(
        Path(output),
        json.dumps(
            report, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
        )
        + "\n",
    )
    if not passed:
        raise RuntimeError(f"CPU regression gate failed; inspect {log_path}.")
    return report


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"Non-finite JSON constant: {value}")

    value = json.loads(
        Path(path).read_text(encoding="utf-8"), parse_constant=reject_constant
    )
    if not isinstance(value, dict):
        raise RuntimeError("CPU gate report must be a JSON object.")
    return value


def validate_cpu_gate_report(
    signature: Mapping[str, Any],
    path: Path,
    *,
    expected_log: Path,
) -> dict[str, Any]:
    """Reject stale, partial, forged, or non-finite CPU evidence."""

    report = _strict_json(path)
    required = {
        "schema_version",
        "gate",
        "passed",
        "runtime_signature_sha256",
        "suite_order",
        "suites",
        "total_tests",
        "elapsed_seconds",
        "log_path",
        "log_sha256",
    }
    if set(report) != required:
        raise RuntimeError("CPU gate report has an invalid field set.")
    if (
        report["schema_version"] != CPU_GATE_SCHEMA_VERSION
        or report["gate"] != "cpu"
        or report["passed"] is not True
        or report["runtime_signature_sha256"] != signature["sha256"]
        or report["suite_order"] != list(CPU_SUITES)
    ):
        raise RuntimeError("CPU gate report is stale or did not pass.")
    suites = report["suites"]
    if not isinstance(suites, list) or len(suites) != len(CPU_SUITES):
        raise RuntimeError("CPU gate report does not contain every suite.")
    expected_record_keys = {
        "suite",
        "returncode",
        "test_count",
        "test_file_count",
        "elapsed_seconds",
    }
    total = 0
    root = _project_root()
    for expected_name, record in zip(CPU_SUITES, suites, strict=True):
        if not isinstance(record, dict) or set(record) != expected_record_keys:
            raise RuntimeError("CPU suite report has an invalid field set.")
        expected_files = len(tuple((root / "tests" / expected_name).glob("test_*.py")))
        elapsed = record["elapsed_seconds"]
        if (
            record["suite"] != expected_name
            or record["returncode"] != 0
            or isinstance(record["test_count"], bool)
            or not isinstance(record["test_count"], int)
            or record["test_count"] <= 0
            or record["test_file_count"] != expected_files
            or not isinstance(elapsed, (int, float))
            or isinstance(elapsed, bool)
            or not math.isfinite(float(elapsed))
            or float(elapsed) <= 0.0
        ):
            raise RuntimeError(f"CPU suite evidence is invalid: {expected_name}.")
        total += record["test_count"]
    elapsed = report["elapsed_seconds"]
    if (
        report["total_tests"] != total
        or not isinstance(elapsed, (int, float))
        or isinstance(elapsed, bool)
        or not math.isfinite(float(elapsed))
        or float(elapsed) <= 0.0
    ):
        raise RuntimeError("CPU gate totals are invalid.")
    expected_log_path = Path(expected_log).expanduser().resolve()
    if (
        report["log_path"] != str(expected_log_path)
        or not expected_log_path.is_file()
        or report["log_sha256"] != _sha256_file(expected_log_path)
    ):
        raise RuntimeError("CPU gate log is missing or has changed.")
    return report


__all__ = [
    "CPU_GATE_SCHEMA_VERSION",
    "CPU_SUITES",
    "run_cpu_gate",
    "validate_cpu_gate_report",
]
