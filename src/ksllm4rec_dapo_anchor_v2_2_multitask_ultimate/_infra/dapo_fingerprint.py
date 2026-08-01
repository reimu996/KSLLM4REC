"""Content signature binding every input used by the isolated DAPO-style run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .rloo_fingerprint import frozen_model_inputs, software_versions
from .rloo_integrity import canonical_sha256, file_record, snapshot_directory

from . import dapo_contract as contract
from .. import contract as v2_contract


_PACKAGE_ROOT = Path(__file__).resolve().parent
# Vendored into ksllm4rec_dapo_anchor_v2_2/_infra/ — one deeper than the original
# ksllm4rec_rloo_dapo/, so parents[2] instead of parents[1].
_PROJECT_ROOT = _PACKAGE_ROOT.parents[2]


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
    package_root = root / "src" / "ksllm4rec_dapo_anchor_v2_2_multitask"
    if package_root.is_dir():
        paths.extend(package_root.rglob("*.py"))
    script_root = (
        root
        / "scripts/dapo_anchor/frontier_sft372_dapo_anchor_v2_2_multitask"
    )
    if script_root.is_dir():
        paths.extend(
            path
            for path in script_root.rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".sh"}
            and path.name != "status.sh"
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
    recommendation_groups_dir = Path(
        config["output"]["recommendation_groups_dir"]
    )
    text_to_sid_groups_dir = Path(config["output"]["text_to_sid_groups_dir"])
    trie_dir = Path(config["output"]["trie_dir"])
    config_record: dict[str, Any] = {
        "resolved": config,
        "resolved_sha256": canonical_sha256(config),
    }
    if config_path is not None:
        config_record["source_file"] = _file(Path(config_path))
    inputs = {
        "schema_version": 2,
        "config": config_record,
        "frozen_model_inputs": frozen_model_inputs(config),
        "source": _file(Path(config["data"]["source"])),
        "provenance": _file(Path(config["data"]["provenance"])),
        "groups": {
            "recommendation": _file(recommendation_groups_dir / "groups.jsonl"),
            "item_text_to_sid": _file(text_to_sid_groups_dir / "groups.jsonl"),
        },
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
        **(
            {
                "execution_target": {
                    "device": contract.EXECUTION_DEVICE,
                    "cuda_visible_devices": (contract.EXECUTION_CUDA_VISIBLE_DEVICES),
                    "gpu_identity": dict(contract.EXPECTED_GPU_IDENTITY),
                }
            }
            if config.get("profile") == v2_contract.PROFILE
            else {}
        ),
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
