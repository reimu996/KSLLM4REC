"""Content signature binding every input used by the isolated DAPO-style run."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .rloo_fingerprint import frozen_model_inputs, software_versions
from .rloo_integrity import canonical_sha256, file_record, snapshot_directory

from .. import contract


_PACKAGE_ROOT = Path(__file__).resolve().parent
_PROJECT_ROOT = _PACKAGE_ROOT.parents[2]
_RUNTIME_ROOTS = (
    Path("src/ksllm4rec_dapo_anchor_multitask_v1_1"),
    Path("scripts/dapo_anchor/frontier_sft372_dapo_anchor_multitask_v1_1_e2"),
    Path("tests/dapo_anchor_multitask_v1_1"),
)


def _is_allowed_runtime_path(path: Path, project_root: Path) -> bool:
    return any(
        path.is_relative_to(project_root / relative) for relative in _RUNTIME_ROOTS
    )


def _file(path: Path) -> dict[str, Any]:
    record = file_record(Path(path))
    return {
        "path": record["path"],
        "size": record["size"],
        "sha256": record["sha256"],
    }


def _groups_artifact(path: Path) -> dict[str, Any]:
    directory = Path(path)
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    return {
        "path": str(directory.resolve()),
        "groups": _file(directory / "groups.jsonl"),
        "manifest": _file(directory / "data_manifest.json"),
    }


def _assert_frozen_input_identity(inputs: dict[str, Any]) -> None:
    """Reject a self-consistent signature built from the wrong frozen bytes."""

    expected_files = {
        "source": contract.SOURCE_DATA_SHA256,
        "provenance": contract.PROVENANCE_SHA256,
        "fixed_probe": contract.FIXED_PROBE_SHA256,
    }
    for key, expected in expected_files.items():
        actual = inputs[key]["sha256"]
        if actual != expected:
            raise RuntimeError(
                f"Frozen input {key} fingerprint mismatch: {actual} != {expected}."
            )
    expected_groups = {
        "recommendation_groups": (
            contract.RECOMMENDATION_GROUPS_SHA256,
            contract.RECOMMENDATION_MANIFEST_SHA256,
        ),
        "text_to_sid_groups": (
            contract.TEXT_TO_SID_GROUPS_SHA256,
            contract.TEXT_TO_SID_MANIFEST_SHA256,
        ),
    }
    for key, (groups_sha, manifest_sha) in expected_groups.items():
        if inputs[key]["groups"]["sha256"] != groups_sha:
            raise RuntimeError(f"Frozen input {key}.groups fingerprint mismatch.")
        if inputs[key]["manifest"]["sha256"] != manifest_sha:
            raise RuntimeError(f"Frozen input {key}.manifest fingerprint mismatch.")
    trie_manifest = inputs["trie"]["files"].get("manifest.json")
    if not isinstance(trie_manifest, dict) or (
        trie_manifest.get("sha256") != contract.TRIE_MANIFEST_SHA256
    ):
        raise RuntimeError("Frozen input trie manifest fingerprint mismatch.")
    model_inputs = inputs["frozen_model_inputs"]
    expected_trees = {
        "base_model": contract.BASE_MODEL_TREE_SHA256,
        "sft_adapter": contract.SFT_ADAPTER_TREE_SHA256,
        "tokenizer": contract.TOKENIZER_TREE_SHA256,
    }
    for key, expected in expected_trees.items():
        actual = canonical_sha256(model_inputs[key]["files"])
        if actual != expected:
            raise RuntimeError(
                f"Frozen model input {key} fingerprint mismatch: {actual} != {expected}."
            )


def discover_runtime_code_files(
    project_root: Path = _PROJECT_ROOT,
) -> tuple[Path, ...]:
    root = Path(project_root).resolve()
    paths: list[Path] = []
    package_root = root / _RUNTIME_ROOTS[0]
    if package_root.is_dir():
        paths.extend(package_root.rglob("*.py"))
    script_root = root / _RUNTIME_ROOTS[1]
    if script_root.is_dir():
        paths.extend(
            path
            for path in script_root.rglob("*")
            if path.is_file()
            and path.suffix in {".py", ".sh"}
        )
    suite_root = root / _RUNTIME_ROOTS[2]
    if suite_root.is_dir():
        paths.extend(suite_root.rglob("*.py"))
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
        if not _is_allowed_runtime_path(path, root):
            raise RuntimeError(
                f"Runtime code is outside Multitask V1 roots: {path}"
            )
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
    enforce_frozen_identity: bool = True,
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
        "schema_version": 3,
        "config": config_record,
        "frozen_model_inputs": frozen_model_inputs(config),
        "source": _file(Path(config["data"]["source"])),
        "provenance": _file(Path(config["data"]["provenance"])),
        "recommendation_groups": _groups_artifact(recommendation_groups_dir),
        "text_to_sid_groups": _groups_artifact(text_to_sid_groups_dir),
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
            if config.get("profile") == contract.PROFILE
            else {}
        ),
        "runtime_code": runtime_code_fingerprint(
            project_root, runtime_code_files=runtime_code_files
        ),
    }
    if enforce_frozen_identity:
        _assert_frozen_input_identity(inputs)
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
