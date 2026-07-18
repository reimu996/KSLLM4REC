"""Verify immutable inputs used by the reproducible SFT run."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from copy import deepcopy
from difflib import unified_diff
from itertools import zip_longest
from pathlib import Path
from typing import Any

from .data import (
    iter_derived_records,
    iter_source_records,
    sha256_file,
    validate_dataset_info,
)
from .manifest import atomic_write_json
from .profiles import MODEL_PATH, SFTProfile, get_profile


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
    source["records"] = lock["source_dataset"].get("records")
    derived = _verify_file(
        Path(lock["derived_dataset"]["path"]).resolve(), lock["derived_dataset"]
    )
    derived["records"] = lock["derived_dataset"].get("records")
    dataset_info = None
    dataset_info_lock = lock.get("dataset_info")
    lock_profile = lock.get("profile")
    if dataset_info_lock is not None:
        dataset_info_path = Path(dataset_info_lock["path"]).resolve()
        dataset_info = _verify_file(dataset_info_path, dataset_info_lock)
        if lock_profile is not None:
            selected = get_profile(str(lock_profile))
            validate_dataset_info(
                dataset_info_path,
                dataset_name=selected.dataset_name,
                derived_filename=selected.derived_path.name,
            )
    elif lock_profile not in (None, "baseline"):
        raise RuntimeError(
            f"Artifact lock for profile {lock_profile!r} must include dataset_info."
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
        "profile": lock_profile,
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "model_root": str(model_root),
        "model_files": model_files,
        "source_dataset": source,
        "derived_dataset": derived,
        "dataset_info": dataset_info,
        "llamafactory": {
            "path": str(llamafactory_root),
            "commit": actual_commit,
            "tree": actual_tree,
            "working_tree_clean": True,
            "imported_from": str(imported_from) if imported_from else None,
        },
    }


def artifact_identity(report: dict[str, Any]) -> dict[str, Any]:
    """Return the immutable subset used to bind manifests to one input lock."""

    model_files = {
        name: {
            "path": value.get("path"),
            "size": value.get("size"),
            "sha256": value.get("sha256"),
        }
        for name, value in sorted(report.get("model_files", {}).items())
    }

    def file_identity(name: str) -> dict[str, Any]:
        value = report.get(name) or {}
        return {
            "path": value.get("path"),
            "size": value.get("size"),
            "sha256": value.get("sha256"),
            "records": value.get("records"),
        }

    llamafactory = report.get("llamafactory", {})
    return {
        "profile": report.get("profile"),
        "lock_path": report.get("lock_path"),
        "lock_sha256": report.get("lock_sha256"),
        "model_root": report.get("model_root"),
        "model_files": model_files,
        "source_dataset": file_identity("source_dataset"),
        "derived_dataset": file_identity("derived_dataset"),
        "dataset_info": file_identity("dataset_info"),
        "llamafactory": {
            "path": llamafactory.get("path"),
            "commit": llamafactory.get("commit"),
            "tree": llamafactory.get("tree"),
        },
    }


def require_matching_artifact_identity(
    actual: dict[str, Any], expected: dict[str, Any]
) -> dict[str, Any]:
    actual_identity = artifact_identity(actual)
    expected_identity = artifact_identity(expected)
    if actual_identity != expected_identity:
        raise RuntimeError(
            "Artifact lock identity mismatch: "
            f"expected {expected_identity!r}, got {actual_identity!r}"
        )
    return actual_identity


def validate_profile_artifact_identity(
    report: dict[str, Any], profile: str | SFTProfile
) -> dict[str, Any]:
    """Reject a valid lock when it belongs to a different approved data profile."""

    selected = get_profile(profile)
    source = report.get("source_dataset", {})
    derived = report.get("derived_dataset", {})
    checks = {
        "lock_path": (
            report.get("lock_path"),
            str(selected.artifact_lock_path.resolve()),
        ),
        "model_root": (report.get("model_root"), str(MODEL_PATH.resolve())),
        "source.path": (source.get("path"), str(selected.source_path.resolve())),
        "source.size": (source.get("size"), selected.source_size),
        "source.sha256": (source.get("sha256"), selected.source_sha256),
        "source.records": (source.get("records"), selected.source_records),
        "derived.path": (derived.get("path"), str(selected.derived_path.resolve())),
        "derived.records": (derived.get("records"), selected.source_records),
    }
    dataset_info = report.get("dataset_info")
    if selected.name != "baseline" or dataset_info is not None:
        dataset_info = dataset_info or {}
        checks["dataset_info.path"] = (
            dataset_info.get("path"),
            str((selected.dataset_dir / "dataset_info.json").resolve()),
        )
    lock_profile = report.get("profile")
    if selected.name == "baseline":
        if lock_profile not in (None, selected.name):
            checks["profile"] = (lock_profile, selected.name)
    else:
        checks["profile"] = (lock_profile, selected.name)
    mismatches = [
        f"{name}: expected {expected!r}, got {actual!r}"
        for name, (actual, expected) in checks.items()
        if actual != expected
    ]
    if mismatches:
        raise RuntimeError(
            f"Artifact lock does not belong to SFT profile {selected.name!r}:\n- "
            + "\n- ".join(mismatches)
        )
    return {"profile": selected.name, "identity": artifact_identity(report)}


def create_profile_artifact_lock(
    *,
    profile: str | SFTProfile,
    source_path: Path,
    derived_path: Path,
    base_lock_path: Path,
    output_path: Path,
    llamafactory_module_file: Path | None = None,
) -> dict[str, Any]:
    """Create a data-specific lock without allowing arbitrary source promotion."""

    selected = get_profile(profile)
    source_path = source_path.resolve()
    derived_path = derived_path.resolve()
    if source_path != selected.source_path.resolve():
        raise RuntimeError(
            f"Source path for profile {selected.name!r} must be "
            f"{selected.source_path.resolve()}, got {source_path}"
        )
    if derived_path != selected.derived_path.resolve():
        raise RuntimeError(
            f"Derived path for profile {selected.name!r} must be "
            f"{selected.derived_path.resolve()}, got {derived_path}"
        )
    dataset_info_path = derived_path.parent / "dataset_info.json"
    validate_dataset_info(
        dataset_info_path,
        dataset_name=selected.dataset_name,
        derived_filename=derived_path.name,
    )
    if base_lock_path.resolve() != get_profile("baseline").artifact_lock_path.resolve():
        raise RuntimeError(
            "Base artifact lock must be the approved baseline lock: "
            f"expected {get_profile('baseline').artifact_lock_path.resolve()}, "
            f"got {base_lock_path.resolve()}"
        )
    if output_path.resolve() != selected.artifact_lock_path.resolve():
        raise RuntimeError(
            f"Output lock for profile {selected.name!r} must be "
            f"{selected.artifact_lock_path.resolve()}, got {output_path.resolve()}"
        )

    source_size = source_path.stat().st_size
    source_sha256 = sha256_file(source_path)
    source_records = 0
    derived_records = 0
    sentinel = object()
    for line_number, (source_record, derived_record) in enumerate(
        zip_longest(
            iter_source_records(source_path),
            iter_derived_records(derived_path),
            fillvalue=sentinel,
        ),
        start=1,
    ):
        if source_record is sentinel or derived_record is sentinel:
            raise RuntimeError(
                "Derived data record count differs from the approved source at "
                f"line {line_number}."
            )
        source_records += 1
        derived_records += 1
        if source_record != derived_record:
            raise RuntimeError(
                f"Derived data does not preserve source content at line {line_number}."
            )
    source_checks = {
        "size": (source_size, selected.source_size),
        "sha256": (source_sha256, selected.source_sha256),
        "records": (source_records, selected.source_records),
    }
    source_mismatches = [
        f"{name}: expected {expected!r}, got {actual!r}"
        for name, (actual, expected) in source_checks.items()
        if actual != expected
    ]
    if source_mismatches:
        raise RuntimeError(
            f"Approved source identity mismatch for profile {selected.name!r}:\n- "
            + "\n- ".join(source_mismatches)
        )

    if derived_records != selected.source_records:
        raise RuntimeError(
            f"Derived record count mismatch: expected {selected.source_records}, "
            f"got {derived_records}"
        )

    base_report = verify_artifact_lock(
        base_lock_path, llamafactory_module_file=llamafactory_module_file
    )
    validate_profile_artifact_identity(base_report, get_profile("baseline"))
    base_lock = json.loads(base_lock_path.read_text(encoding="utf-8"))
    lock = {
        "schema_version": 1,
        "profile": selected.name,
        "llamafactory": deepcopy(base_lock["llamafactory"]),
        "model": deepcopy(base_lock["model"]),
        "source_dataset": {
            "path": str(source_path),
            "records": source_records,
            "sha256": source_sha256,
            "size": source_size,
        },
        "derived_dataset": {
            "path": str(derived_path),
            "records": derived_records,
            "sha256": sha256_file(derived_path),
            "size": derived_path.stat().st_size,
        },
        "dataset_info": {
            "path": str(dataset_info_path.resolve()),
            "sha256": sha256_file(dataset_info_path),
            "size": dataset_info_path.stat().st_size,
        },
    }
    atomic_write_json(output_path, lock)
    return {
        "status": "passed",
        "profile": selected.name,
        "output_path": str(output_path.resolve()),
        "base_lock_identity": artifact_identity(base_report),
        "source_dataset": lock["source_dataset"],
        "derived_dataset": lock["derived_dataset"],
        "dataset_info": lock["dataset_info"],
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
