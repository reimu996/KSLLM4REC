"""Runtime contract fingerprints shared by gates, recovery, and probes."""

from __future__ import annotations

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path
from typing import Any, Iterable

from .contract import (
    BASE_MODEL_SHA256,
    GROUPS_SHA256,
    SFT_ADAPTER_SHA256,
    SFT_CONFIG_SHA256,
    TRIE_MANIFEST_SHA256,
)
from .integrity import require_file, sha256_file


_PACKAGE_ROOT = Path(__file__).resolve().parent
_SOURCE_ROOT = _PACKAGE_ROOT.parent

_RUNTIME_CODE_FILES = (
    "ksllm4rec_grpo/checkpoint.py",
    "ksllm4rec_grpo/cli.py",
    "ksllm4rec_grpo/config.py",
    "ksllm4rec_grpo/constraint.py",
    "ksllm4rec_grpo/contract.py",
    "ksllm4rec_grpo/data.py",
    "ksllm4rec_grpo/fingerprint.py",
    "ksllm4rec_grpo/gates.py",
    "ksllm4rec_grpo/integrity.py",
    "ksllm4rec_grpo/modeling.py",
    "ksllm4rec_grpo/objective.py",
    "ksllm4rec_grpo/probability.py",
    "ksllm4rec_grpo/prompt.py",
    "ksllm4rec_grpo/rollout.py",
    "ksllm4rec_grpo/scoring.py",
    "ksllm4rec_grpo/trainer.py",
    "ksllm4rec_grpo/trie.py",
    "ksllm4rec_orpo/data.py",
    "ksllm4rec_sft/data.py",
)

_PROBE_CODE_FILES = (
    "ksllm4rec_grpo/constraint.py",
    "ksllm4rec_grpo/fingerprint.py",
    "ksllm4rec_grpo/integrity.py",
    "ksllm4rec_grpo/modeling.py",
    "ksllm4rec_grpo/probability.py",
    "ksllm4rec_grpo/probe.py",
    "ksllm4rec_grpo/prompt.py",
    "ksllm4rec_grpo/trie.py",
    "ksllm4rec_orpo/data.py",
    "ksllm4rec_sft/data.py",
)


def _file_hashes(relative_paths: Iterable[str]) -> dict[str, str]:
    return {
        relative: sha256_file(_SOURCE_ROOT / relative) for relative in relative_paths
    }


def software_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "torch": version("torch"),
        "transformers": version("transformers"),
        "peft": version("peft"),
        "flash-attn": version("flash-attn"),
        "tokenizers": version("tokenizers"),
        "accelerate": version("accelerate"),
        "safetensors": version("safetensors"),
        "numpy": version("numpy"),
        "triton": version("triton"),
    }


def frozen_model_inputs(config: dict[str, Any]) -> dict[str, Any]:
    base = Path(config["model"]["base_model"])
    adapter = Path(config["model"]["sft_adapter"])
    tokenizer = Path(config["model"]["tokenizer"])
    records = {
        "base_weights": require_file(base / "model.safetensors", BASE_MODEL_SHA256),
        "sft_adapter": require_file(
            adapter / "adapter_model.safetensors", SFT_ADAPTER_SHA256
        ),
        "sft_adapter_config": require_file(
            adapter / "adapter_config.json", SFT_CONFIG_SHA256
        ),
    }
    ancillary = {
        "base_config": base / "config.json",
        "base_weight_index": base / "model.safetensors.index.json",
        "base_generation_config": base / "generation_config.json",
        "tokenizer": tokenizer / "tokenizer.json",
        "tokenizer_config": tokenizer / "tokenizer_config.json",
        "tokenizer_added_tokens": tokenizer / "added_tokens.json",
        "tokenizer_special_tokens": tokenizer / "special_tokens_map.json",
        "tokenizer_chat_template": tokenizer / "chat_template.jinja",
        "tokenizer_merges": tokenizer / "merges.txt",
        "tokenizer_vocab": tokenizer / "vocab.json",
    }
    for name, path in ancillary.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        records[name] = {
            "path": str(path.resolve()),
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return records


def _signature(inputs: dict[str, Any]) -> dict[str, Any]:
    canonical = json.dumps(inputs, sort_keys=True, separators=(",", ":")).encode()
    return {"sha256": hashlib.sha256(canonical).hexdigest(), "inputs": inputs}


def runtime_signature(
    config: dict[str, Any], groups_path: Path, trie_dir: Path
) -> dict[str, Any]:
    groups_record = require_file(Path(groups_path), GROUPS_SHA256)
    trie_record = require_file(Path(trie_dir) / "manifest.json", TRIE_MANIFEST_SHA256)
    inputs = {
        "config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "groups": groups_record,
        "trie_manifest": trie_record,
        "frozen_model_inputs": frozen_model_inputs(config),
        "software_versions": software_versions(),
        "module_sha256": _file_hashes(_RUNTIME_CODE_FILES),
    }
    return _signature(inputs)


def probe_algorithm_inputs(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "frozen_model_inputs": frozen_model_inputs(config),
        "software_versions": software_versions(),
        "module_sha256": _file_hashes(_PROBE_CODE_FILES),
    }
