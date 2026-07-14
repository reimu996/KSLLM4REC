"""Verify immutable inputs used by the reproducible SFT run."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from difflib import unified_diff
from pathlib import Path
from typing import Any

from .data import sha256_file


EXPECTED_PYTHON_VERSION = "3.11.15"
EXPECTED_CONDA_PREFIX = Path("/home/lyc/miniconda3/envs/onereason_lora_sft")
OUTPUT_ARTIFACT_NAMES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "trainer_state.json",
    "train_results.json",
)


def _verify_file(path: Path, expected: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Locked artifact is missing: {path}")
    actual_size = path.stat().st_size
    expected_size = int(expected["size"])
    if actual_size != expected_size:
        raise RuntimeError(
            f"Size mismatch for {path}: expected {expected_size}, got {actual_size}"
        )
    actual_sha256 = sha256_file(path)
    expected_sha256 = str(expected["sha256"])
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f"SHA256 mismatch for {path}: expected {expected_sha256}, got {actual_sha256}"
        )
    return {"path": str(path), "size": actual_size, "sha256": actual_sha256}


def verify_artifact_lock(
    lock_path: Path, *, llamafactory_module_file: Path | None = None
) -> dict[str, Any]:
    """Return verified input metadata, or raise before training can start."""

    lock_path = lock_path.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1:
        raise RuntimeError(
            f"Unsupported artifact lock schema: {lock.get('schema_version')!r}"
        )

    model = lock["model"]
    model_root = Path(model["path"]).resolve()
    model_files = {
        name: _verify_file(model_root / name, expected)
        for name, expected in sorted(model["files"].items())
    }
    source = _verify_file(
        Path(lock["source_dataset"]["path"]).resolve(), lock["source_dataset"]
    )
    derived = _verify_file(
        Path(lock["derived_dataset"]["path"]).resolve(), lock["derived_dataset"]
    )

    llamafactory = lock["llamafactory"]
    llamafactory_root = Path(llamafactory["path"]).resolve()
    result = subprocess.run(
        ["git", "rev-parse", "HEAD", "HEAD^{tree}"],
        cwd=llamafactory_root,
        check=True,
        capture_output=True,
        text=True,
    )
    identity_lines = result.stdout.splitlines()
    if len(identity_lines) != 2:
        raise RuntimeError(
            f"Unexpected LLaMA-Factory git identity output: {result.stdout!r}"
        )
    actual_commit, actual_tree = identity_lines
    expected_commit = str(llamafactory["commit"])
    if actual_commit != expected_commit:
        raise RuntimeError(
            f"LLaMA-Factory commit mismatch: expected {expected_commit}, got {actual_commit}"
        )
    expected_tree = str(llamafactory["tree"])
    if actual_tree != expected_tree:
        raise RuntimeError(
            f"LLaMA-Factory tree mismatch: expected {expected_tree}, got {actual_tree}"
        )
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=llamafactory_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if status.stdout:
        raise RuntimeError(
            "LLaMA-Factory working tree is not clean:\n" + status.stdout.rstrip()
        )
    imported_from = None
    if llamafactory_module_file is not None:
        imported_from = llamafactory_module_file.resolve()
        expected_package = (llamafactory_root / "src" / "llamafactory").resolve()
        if not imported_from.is_relative_to(expected_package):
            raise RuntimeError(
                "Imported LLaMA-Factory does not come from the locked source tree: "
                f"imported={imported_from}, expected_under={expected_package}"
            )

    return {
        "status": "passed",
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "model_root": str(model_root),
        "model_files": model_files,
        "source_dataset": source,
        "derived_dataset": derived,
        "llamafactory": {
            "path": str(llamafactory_root),
            "commit": actual_commit,
            "tree": actual_tree,
            "working_tree_clean": True,
            "imported_from": str(imported_from) if imported_from else None,
        },
    }


def verify_environment_lock(lock_path: Path) -> dict[str, Any]:
    lock_path = lock_path.resolve()
    actual_prefix = Path(sys.prefix).resolve()
    actual_executable = Path(sys.executable).resolve()
    expected_bin = (EXPECTED_CONDA_PREFIX / "bin").resolve()
    if (
        actual_prefix != EXPECTED_CONDA_PREFIX
        or actual_executable.parent != expected_bin
    ):
        raise RuntimeError(
            "Conda environment mismatch: "
            f"expected prefix {EXPECTED_CONDA_PREFIX}, got prefix {actual_prefix} "
            f"and executable {actual_executable}"
        )
    actual_python = platform.python_version()
    if actual_python != EXPECTED_PYTHON_VERSION:
        raise RuntimeError(
            f"Python version mismatch: expected {EXPECTED_PYTHON_VERSION}, got {actual_python}"
        )
    expected = lock_path.read_text(encoding="utf-8").splitlines()
    result = subprocess.run(
        [sys.executable, "-m", "pip", "freeze"],
        check=True,
        capture_output=True,
        text=True,
    )
    actual = result.stdout.splitlines()
    if actual != expected:
        difference = "\n".join(
            unified_diff(
                expected, actual, fromfile=str(lock_path), tofile="pip freeze", n=3
            )
        )
        raise RuntimeError(
            f"Installed environment does not match the lock file:\n{difference}"
        )
    return {
        "status": "passed",
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "packages": len(actual),
        "python_version": actual_python,
        "python_executable": str(actual_executable),
    }


def snapshot_output_artifacts(output_dir: Path) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    artifacts = {}
    for name in OUTPUT_ARTIFACT_NAMES:
        path = output_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"Required training output is missing: {path}")
        artifacts[name] = {
            "path": str(path),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return artifacts


def verify_output_artifacts(
    output_dir: Path, expected_artifacts: dict[str, Any]
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    if set(expected_artifacts) != set(OUTPUT_ARTIFACT_NAMES):
        raise RuntimeError(
            "Manifest output artifact set mismatch: "
            f"expected {list(OUTPUT_ARTIFACT_NAMES)!r}, got {sorted(expected_artifacts)!r}"
        )
    verified = {}
    for name in OUTPUT_ARTIFACT_NAMES:
        expected = expected_artifacts[name]
        expected_path = Path(expected.get("path", "")).resolve()
        actual_path = output_dir / name
        if expected_path != actual_path:
            raise RuntimeError(
                f"Manifest path mismatch for {name}: expected {actual_path}, got {expected_path}"
            )
        verified[name] = _verify_file(actual_path, expected)
    return verified
