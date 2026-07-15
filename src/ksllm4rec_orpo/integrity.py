"""Immutable source and training-artifact locks for the ORPO run."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Iterable

from ksllm4rec_sft.data import sha256_file
from ksllm4rec_sft.manifest import atomic_write_json


MODEL_FILES = (
    "config.json",
    "model.safetensors",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _file_entry(path: Path, *, root: Path | None = None) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Locked input is missing: {path}")
    return {
        "path": str(path.relative_to(root.resolve()) if root else path),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _directory_entry(
    directory: Path, pattern: str, *, expected_files: int
) -> dict[str, Any]:
    directory = directory.resolve()
    files = sorted(path for path in directory.glob(pattern) if path.is_file())
    if len(files) != expected_files:
        raise RuntimeError(
            f"Expected {expected_files} files matching {pattern!r} under "
            f"{directory}, found {len(files)}."
        )
    return {
        "path": str(directory),
        "pattern": pattern,
        "files": [_file_entry(path, root=directory) for path in files],
    }


def _git_identity(repository: Path) -> dict[str, Any]:
    repository = repository.resolve()
    result = subprocess.run(
        ["git", "rev-parse", "HEAD", "HEAD^{tree}"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    )
    commit, tree = result.stdout.splitlines()
    status = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=repository,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status:
        raise RuntimeError(f"Locked repository is not clean: {repository}\n{status}")
    return {
        "path": str(repository),
        "commit": commit,
        "tree": tree,
        "working_tree_clean": True,
    }


def create_generation_lock(
    lock_path: Path,
    *,
    model_root: Path,
    source_dataset: Path,
    pid2sid_dir: Path,
    explorer_caption_dir: Path,
    explorer_text_probe_dir: Path,
    explorer_recommend_probe_dir: Path,
    llamafactory_root: Path,
) -> dict[str, Any]:
    """Hash every raw input used to construct pairs and fixed probes."""

    lock_path = lock_path.resolve()
    if lock_path.exists():
        raise FileExistsError(f"Generation lock already exists: {lock_path}")
    model_root = model_root.resolve()
    value = {
        "schema_version": 1,
        "kind": "orpo_generation_sources",
        "model": {
            "path": str(model_root),
            "files": {name: _file_entry(model_root / name) for name in MODEL_FILES},
        },
        "source_dataset": _file_entry(source_dataset),
        "pid2sid": _directory_entry(pid2sid_dir, "*.parquet", expected_files=198),
        "explorer_caption": _directory_entry(
            explorer_caption_dir, "*.jsonl", expected_files=31
        ),
        "explorer_text_probe": _directory_entry(
            explorer_text_probe_dir, "*.jsonl", expected_files=31
        ),
        "explorer_recommend_probe": _directory_entry(
            explorer_recommend_probe_dir, "*.jsonl", expected_files=6
        ),
        "llamafactory": _git_identity(llamafactory_root),
    }
    atomic_write_json(lock_path, value)
    return value


def _verify_file_entry(entry: dict[str, Any], *, root: Path | None = None) -> dict[str, Any]:
    path = Path(entry["path"])
    if root is not None:
        path = root / path
    actual = _file_entry(path)
    if actual["size"] != int(entry["size"]) or actual["sha256"] != entry["sha256"]:
        raise RuntimeError(f"Locked file changed: {path.resolve()}")
    return actual


def _verify_directory_entry(entry: dict[str, Any]) -> dict[str, Any]:
    root = Path(entry["path"]).resolve()
    actual_paths = sorted(path for path in root.glob(entry["pattern"]) if path.is_file())
    expected_names = [item["path"] for item in entry["files"]]
    actual_names = [str(path.relative_to(root)) for path in actual_paths]
    if actual_names != expected_names:
        raise RuntimeError(f"Locked directory membership changed: {root}")
    for item in entry["files"]:
        _verify_file_entry(item, root=root)
    return {"path": str(root), "files": len(actual_paths)}


def _verify_git(entry: dict[str, Any], imported_module: Path | None) -> dict[str, Any]:
    actual = _git_identity(Path(entry["path"]))
    if actual["commit"] != entry["commit"] or actual["tree"] != entry["tree"]:
        raise RuntimeError("LLaMA-Factory git identity changed after locking.")
    if imported_module is not None:
        expected = (Path(entry["path"]) / "src" / "llamafactory").resolve()
        if not imported_module.resolve().is_relative_to(expected):
            raise RuntimeError(
                f"Imported LLaMA-Factory is outside locked tree: {imported_module}"
            )
        actual["imported_from"] = str(imported_module.resolve())
    return actual


def verify_generation_lock(lock_path: Path) -> dict[str, Any]:
    lock_path = lock_path.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1 or lock.get("kind") != "orpo_generation_sources":
        raise RuntimeError("Unsupported ORPO generation lock.")
    model_root = Path(lock["model"]["path"]).resolve()
    for entry in lock["model"]["files"].values():
        _verify_file_entry(entry)
    _verify_file_entry(lock["source_dataset"])
    directories = {
        name: _verify_directory_entry(lock[name])
        for name in (
            "pid2sid",
            "explorer_caption",
            "explorer_text_probe",
            "explorer_recommend_probe",
        )
    }
    git = _verify_git(lock["llamafactory"], None)
    return {
        "status": "passed",
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "model_root": str(model_root),
        "directories": directories,
        "llamafactory": git,
    }


def create_training_lock(
    lock_path: Path,
    *,
    generation_lock: Path,
    model_root: Path,
    source_dataset: Path,
    data_files: Iterable[Path],
    llamafactory_root: Path,
) -> dict[str, Any]:
    lock_path = lock_path.resolve()
    if lock_path.exists():
        raise FileExistsError(f"Training lock already exists: {lock_path}")
    model_root = model_root.resolve()
    files = [path.resolve() for path in data_files]
    value = {
        "schema_version": 1,
        "kind": "orpo_training_inputs",
        "generation_lock": _file_entry(generation_lock),
        "model": {
            "path": str(model_root),
            "files": {name: _file_entry(model_root / name) for name in MODEL_FILES},
        },
        "source_dataset": _file_entry(source_dataset),
        "data_files": {path.name: _file_entry(path) for path in files},
        "llamafactory": _git_identity(llamafactory_root),
    }
    if len(value["data_files"]) != len(files):
        raise ValueError("Training data file names must be unique.")
    atomic_write_json(lock_path, value)
    return value


def verify_training_lock(
    lock_path: Path, *, llamafactory_module_file: Path | None = None
) -> dict[str, Any]:
    lock_path = lock_path.resolve()
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("schema_version") != 1 or lock.get("kind") != "orpo_training_inputs":
        raise RuntimeError("Unsupported ORPO training lock.")
    _verify_file_entry(lock["generation_lock"])
    for entry in lock["model"]["files"].values():
        _verify_file_entry(entry)
    source = _verify_file_entry(lock["source_dataset"])
    data_files = {
        name: _verify_file_entry(entry)
        for name, entry in sorted(lock["data_files"].items())
    }
    git = _verify_git(lock["llamafactory"], llamafactory_module_file)
    return {
        "status": "passed",
        "lock_path": str(lock_path),
        "lock_sha256": sha256_file(lock_path),
        "source_dataset": source,
        "data_files": data_files,
        "llamafactory": git,
    }


def snapshot_files(paths: Iterable[Path]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for path in paths:
        resolved = path.resolve()
        if not resolved.is_file():
            raise FileNotFoundError(f"Required output is missing: {resolved}")
        result[resolved.name] = _file_entry(resolved)
    return result
