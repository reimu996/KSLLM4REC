"""Content signature binding every input used by the isolated DAPO-style run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from ksllm4rec_rloo.fingerprint import frozen_model_inputs, software_versions
from ksllm4rec_rloo.integrity import canonical_sha256, file_record, snapshot_directory


_PACKAGE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_ROOT.parents[1]


def _file(path: Path) -> dict[str, Any]:
    record = file_record(Path(path))
    return {
        "path": record["path"],
        "size": record["size"],
        "sha256": record["sha256"],
    }


def discover_runtime_code_files(
    project_root: Path = _PROJECT_ROOT,
) -> tuple[Path, ...]:
    root = Path(project_root).resolve()
    paths: list[Path] = []
    for package in (
        "ksllm4rec_rloo_dapo",
        "ksllm4rec_rloo",
        "ksllm4rec_grpo",
        "ksllm4rec_orpo",
        "ksllm4rec_sft",
    ):
        package_root = root / "src" / package
        if package_root.is_dir():
            paths.extend(package_root.rglob("*.py"))
    script_root = root / "scripts" / "rloo_dapo"
    if script_root.is_dir():
        paths.extend(
            path
            for path in script_root.rglob("*")
            if path.is_file() and path.suffix in {".py", ".sh"}
        )
    result = tuple(sorted({path.resolve() for path in paths}))
    if not result:
        raise RuntimeError("Runtime code fingerprint contains no files.")
    return result


def runtime_code_fingerprint(
    project_root: Path = _PROJECT_ROOT,
    runtime_code_files: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    root = Path(project_root).resolve()
    paths = (
        discover_runtime_code_files(root)
        if runtime_code_files is None
        else tuple(
            (root / path).resolve()
            if not Path(path).is_absolute()
            else Path(path).resolve()
            for path in runtime_code_files
        )
    )
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(set(paths)):
        if not path.is_relative_to(root):
            raise RuntimeError(f"Runtime code is outside the project root: {path}")
        record = file_record(path)
        records[path.relative_to(root).as_posix()] = {
            "size": record["size"],
            "sha256": record["sha256"],
        }
    return {"sha256": canonical_sha256(records), "files": records}


def runtime_signature(
    config: dict[str, Any],
    *,
    config_path: Path | None = None,
    project_root: Path = _PROJECT_ROOT,
    runtime_code_files: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    groups_dir = Path(config["output"]["groups_dir"])
    trie_dir = Path(config["output"]["trie_dir"])
    config_record: dict[str, Any] = {
        "resolved": config,
        "resolved_sha256": canonical_sha256(config),
    }
    if config_path is not None:
        config_record["source_file"] = _file(Path(config_path))
    inputs = {
        "schema_version": 1,
        "config": config_record,
        "frozen_model_inputs": frozen_model_inputs(config),
        "source": _file(Path(config["data"]["source"])),
        "provenance": _file(Path(config["data"]["provenance"])),
        "groups": _file(groups_dir / "groups.jsonl"),
        "trie": {
            "path": str(trie_dir.resolve()),
            "files": snapshot_directory(trie_dir),
        },
        "fixed_probe": _file(Path(config["evaluation"]["fixed_probe"])),
        "software_versions": software_versions(),
        "deterministic_runtime": {
            "cublas_workspace_config": ":4096:8",
            "flash_attention_deterministic": True,
            "torch_deterministic_algorithms": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
        },
        "runtime_code": runtime_code_fingerprint(
            project_root, runtime_code_files=runtime_code_files
        ),
    }
    return {"sha256": canonical_sha256(inputs), "inputs": inputs}


def validate_runtime_signature(signature: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(signature, dict) or set(signature) != {"sha256", "inputs"}:
        raise ValueError("Runtime signature has an invalid schema.")
    actual = canonical_sha256(signature["inputs"])
    if signature["sha256"] != actual:
        raise RuntimeError("Runtime signature is self-inconsistent.")
    return signature


__all__ = [
    "discover_runtime_code_files",
    "runtime_code_fingerprint",
    "runtime_signature",
    "validate_runtime_signature",
]
