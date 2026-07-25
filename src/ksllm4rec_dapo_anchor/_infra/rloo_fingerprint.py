"""Runtime contract fingerprints for the independent RLOO implementation."""

from __future__ import annotations

import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Iterable

from .rloo_integrity import canonical_sha256, file_record, snapshot_directory


_PACKAGE_ROOT = Path(__file__).resolve().parent
# Vendored into ksllm4rec_dapo_anchor/_infra/ — parents[2] hits the project root.
_PROJECT_ROOT = _PACKAGE_ROOT.parents[2]

_BASE_REQUIRED_FILES = ("config.json",)
_BASE_OPTIONAL_FILES = (
    "configuration.json",
    "generation_config.json",
    "model.safetensors.index.json",
)
_ADAPTER_REQUIRED_FILES = (
    "adapter_model.safetensors",
    "adapter_config.json",
)
_TOKENIZER_REQUIRED_FILES = ("tokenizer.json", "tokenizer_config.json")
_TOKENIZER_OPTIONAL_FILES = (
    "added_tokens.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "merges.txt",
    "vocab.json",
)
_SOFTWARE_DISTRIBUTIONS = (
    "torch",
    "transformers",
    "peft",
    "flash-attn",
    "tokenizers",
    "accelerate",
    "safetensors",
    "numpy",
    "triton",
)


def software_versions() -> dict[str, str]:
    """Record installed versions without importing GPU-heavy packages."""

    result = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    for distribution in _SOFTWARE_DISTRIBUTIONS:
        try:
            result[distribution] = version(distribution)
        except PackageNotFoundError:
            result[distribution] = "not-installed"
    return result


def _required_model_path(config: dict[str, Any], key: str) -> Path:
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("RLOO config must contain a model mapping.")
    value = model.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"RLOO config model.{key} must be a non-empty path.")
    return Path(value)


def _selected_file_tree(
    root: Path,
    *,
    required: Iterable[str],
    optional: Iterable[str],
    extra_paths: Iterable[Path] = (),
) -> dict[str, Any]:
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    names: list[str] = []
    for name in required:
        path = root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        names.append(name)
    names.extend(name for name in optional if (root / name).is_file())
    for path in extra_paths:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        names.append(path.relative_to(root).as_posix())
    return {
        "path": str(root.resolve()),
        "files": snapshot_directory(root, relative_paths=sorted(set(names))),
    }


def frozen_model_inputs(config: dict[str, Any]) -> dict[str, Any]:
    """Hash the actual base, r64 adapter, and tokenizer selected by this run.

    No expected digest or adapter identity is imported from the historical GRPO
    profile. The current config path and current file bytes are authoritative.
    """

    base = _required_model_path(config, "base_model")
    adapter = _required_model_path(config, "sft_adapter")
    tokenizer = _required_model_path(config, "tokenizer")
    weight_files = sorted(base.glob("model*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No model*.safetensors files below {base}")
    return {
        "base_model": _selected_file_tree(
            base,
            required=_BASE_REQUIRED_FILES,
            optional=_BASE_OPTIONAL_FILES,
            extra_paths=weight_files,
        ),
        "sft_adapter": _selected_file_tree(
            adapter,
            required=_ADAPTER_REQUIRED_FILES,
            optional=(),
        ),
        "tokenizer": _selected_file_tree(
            tokenizer,
            required=_TOKENIZER_REQUIRED_FILES,
            optional=_TOKENIZER_OPTIONAL_FILES,
        ),
    }


def discover_runtime_code_files(project_root: Path = _PROJECT_ROOT) -> tuple[Path, ...]:
    """Discover every Python module and runnable RLOO script in this checkout."""

    project_root = Path(project_root).resolve()
    package_root = project_root / "src" / "ksllm4rec_rloo"
    if not package_root.is_dir():
        raise FileNotFoundError(package_root)
    paths = list(package_root.rglob("*.py"))
    # Prompt rendering, grammar, trie traversal, and SID parsing are reused
    # from the earlier packages.  They change live rollout behavior, so bind
    # their source bytes to gates and recovery as runtime dependencies.
    for dependency_package in (
        "ksllm4rec_grpo",
        "ksllm4rec_orpo",
        "ksllm4rec_sft",
    ):
        dependency_root = project_root / "src" / dependency_package
        if dependency_root.is_dir():
            paths.extend(dependency_root.rglob("*.py"))
    script_root = project_root / "scripts" / "rloo"
    if script_root.is_dir():
        paths.extend(
            path
            for path in script_root.rglob("*")
            if path.is_file() and path.suffix in {".py", ".sh"}
        )
    unique = tuple(sorted({path.resolve() for path in paths}))
    if not unique:
        raise RuntimeError("RLOO runtime fingerprint contains no code files.")
    return unique


def runtime_code_fingerprint(
    project_root: Path = _PROJECT_ROOT,
    runtime_code_files: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    project_root = Path(project_root).resolve()
    if runtime_code_files is None:
        paths = discover_runtime_code_files(project_root)
    else:
        paths = tuple(
            (project_root / path).resolve()
            if not Path(path).is_absolute()
            else Path(path).resolve()
            for path in runtime_code_files
        )
    records: dict[str, dict[str, Any]] = {}
    for path in sorted(set(paths)):
        if not path.is_relative_to(project_root):
            raise RuntimeError(f"Runtime code is outside the project root: {path}")
        relative = path.relative_to(project_root).as_posix()
        record = file_record(path)
        records[relative] = {
            "size": record["size"],
            "sha256": record["sha256"],
        }
    if not records:
        raise RuntimeError("RLOO runtime fingerprint contains no code files.")
    return {"sha256": canonical_sha256(records), "files": records}


def _groups_file(path: Path) -> Path:
    path = Path(path)
    return path / "groups.jsonl" if path.is_dir() else path


def _input_file_record(path: Path) -> dict[str, Any]:
    record = file_record(path)
    return {
        "path": record["path"],
        "size": record["size"],
        "sha256": record["sha256"],
    }


def runtime_signature(
    config: dict[str, Any],
    groups_path: Path,
    trie_dir: Path,
    calibration_ids_path: Path,
    *,
    config_path: Path | None = None,
    project_root: Path = _PROJECT_ROOT,
    runtime_code_files: Iterable[str | Path] | None = None,
) -> dict[str, Any]:
    """Bind every input that can alter rollout, reward, loss, or recovery."""

    trie_dir = Path(trie_dir)
    if not trie_dir.is_dir():
        raise FileNotFoundError(trie_dir)
    config_record: dict[str, Any] = {
        "resolved_sha256": canonical_sha256(config),
        "resolved": config,
    }
    if config_path is not None:
        config_record["source_file"] = _input_file_record(Path(config_path))
    inputs = {
        "schema_version": 1,
        "config": config_record,
        "frozen_model_inputs": frozen_model_inputs(config),
        "groups": _input_file_record(_groups_file(Path(groups_path))),
        "trie": {
            "path": str(trie_dir.resolve()),
            "files": snapshot_directory(trie_dir),
        },
        "calibration_ids": _input_file_record(Path(calibration_ids_path)),
        "software_versions": software_versions(),
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
        raise RuntimeError(
            f"Runtime signature is self-inconsistent: expected={actual}, "
            f"actual={signature['sha256']}"
        )
    return signature
