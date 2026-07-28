"""Real-model fixed-probe Pass@64 evaluation."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

from . import contract
from ._infra.grpo_constraint import RecommendationGrammar
from ._infra.grpo_prompt import encode_prompt
from ._infra.grpo_trie import SidPrefixTrie
from ._infra.orpo_data import Sid
from ._infra.rloo_modeling import load_policy_model
from .probe64 import run_probe64


PROBE_ROWS = 1_024
_TASK_MODES = {
    "recommend": "probe_recommendation",
    "text_to_sid": "probe_text_to_sid",
}
_PROBE_FIELDS = {
    "instruction",
    "source_file",
    "source_line",
    "source_row_sha256",
    "system",
    "target",
    "target_sid",
    "task",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_fixed_probe(path: Path) -> tuple[dict[str, Any], ...]:
    """Read the exact 1,024-row fixed probe and reject identity drift."""

    rows: list[dict[str, Any]] = []
    identities: set[str] = set()
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict) or set(value) != _PROBE_FIELDS:
                raise ValueError(f"Invalid fixed probe schema at line {line_number}.")
            if value["task"] not in _TASK_MODES:
                raise ValueError(f"Unsupported fixed probe task at line {line_number}.")
            identity = str(value["source_row_sha256"])
            if len(identity) != 64 or identity in identities:
                raise ValueError(f"Invalid fixed probe identity at line {line_number}.")
            if Sid.parse(str(value["target_sid"])).render() != value["target_sid"]:
                raise ValueError(f"Non-canonical target SID at line {line_number}.")
            identities.add(identity)
            rows.append(value)
    if len(rows) != PROBE_ROWS:
        raise ValueError(f"Fixed probe must contain exactly {PROBE_ROWS} rows.")
    task_counts = {
        task: sum(row["task"] == task for row in rows) for task in _TASK_MODES
    }
    if task_counts != {"recommend": 512, "text_to_sid": 512}:
        raise ValueError(f"Fixed probe task counts differ: {task_counts!r}.")
    return tuple(rows)


def _selected_indices(indices: Sequence[int] | None) -> tuple[int, ...]:
    if indices is None:
        return tuple(range(PROBE_ROWS))
    values = tuple(int(value) for value in indices)
    if not values or len(values) != len(set(values)):
        raise ValueError("Probe indices must be non-empty and unique.")
    if tuple(sorted(values)) != values or values[0] < 0 or values[-1] >= PROBE_ROWS:
        raise ValueError("Probe indices must be sorted within [0, 1023].")
    return values


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.name}.", suffix=".tmp", dir=output.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                value,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def run_fixed_probe64(
    config: Mapping[str, Any],
    *,
    output_path: Path,
    device: str | torch.device = "cuda:0",
    policy_adapter_path: Path | None = None,
    indices: Sequence[int] | None = None,
    runtime_signature_sha256: str | None = None,
) -> dict[str, Any]:
    """Evaluate selected fixed rows; a full report contains all 1,024 rows."""

    selected = _selected_indices(indices)
    output = Path(output_path).resolve()
    if output.exists():
        raise FileExistsError(output)
    probe_path = Path(config["evaluation"]["fixed_probe"]).resolve()
    probe_rows = load_fixed_probe(probe_path)
    trie_dir = Path(config["output"]["trie_dir"]).resolve()
    trie = SidPrefixTrie.load(
        trie_dir, expected_leaf_count=int(config["trie"]["unique_sids"])
    )
    bundle = load_policy_model(
        config,
        device=device,
        policy_adapter_path=None if policy_adapter_path is None else Path(policy_adapter_path),
    )
    disable_checkpointing = getattr(bundle.model, "gradient_checkpointing_disable", None)
    if callable(disable_checkpointing):
        disable_checkpointing()
    bundle.model.config.use_cache = True
    bundle.model.eval()
    grammars = {
        task: RecommendationGrammar(bundle.tokenizer, trie, mode=mode)
        for task, mode in _TASK_MODES.items()
    }
    torch_device = torch.device(device)
    if torch_device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(torch_device)
    started = time.monotonic()
    results: list[dict[str, Any]] = []
    with torch.inference_mode():
        for probe_index in selected:
            row = probe_rows[probe_index]
            prompt_ids = encode_prompt(
                bundle.tokenizer,
                str(row["system"]),
                str(row["instruction"]),
                cutoff_len=int(config["data"]["cutoff_len"]),
            )
            input_ids = torch.tensor(
                [prompt_ids], dtype=torch.long, device=torch_device
            )
            attention_mask = torch.ones_like(input_ids)
            result = run_probe64(
                bundle.model,
                input_ids=input_ids,
                attention_mask=attention_mask,
                grammar=grammars[str(row["task"])],
                task=str(row["task"]),
                target_sid=Sid.parse(str(row["target_sid"])),
                max_new_tokens=int(config["evaluation"]["max_completion_length"]),
                num_beams=int(config["evaluation"]["num_candidates"]),
                num_return_sequences=int(config["evaluation"]["num_candidates"]),
                cache_implementation="offloaded",
            )
            results.append(
                {
                    "probe_index": probe_index,
                    "task": str(row["task"]),
                    "source_row_sha256": str(row["source_row_sha256"]),
                    "source_line": int(row["source_line"]),
                    "target_sid": result.target_sid,
                    "prompt_tokens": len(prompt_ids),
                    "candidate_sids": list(result.candidate_sids),
                    "hit_rank": result.hit_rank,
                    "pass_at_64": result.pass_at_64,
                }
            )
            del input_ids, attention_mask
            gc.collect()
            if torch_device.type == "cuda":
                torch.cuda.empty_cache()
    policy_path = Path(
        config["model"]["sft_adapter"]
        if policy_adapter_path is None
        else policy_adapter_path
    ).resolve()
    report = {
        "schema_version": 1,
        "spec_version": contract.SPEC_VERSION,
        "arm": config["experiments"]["active"]["arm"],
        "complete": selected == tuple(range(PROBE_ROWS)),
        "num_candidates": contract.EVALUATION_CANDIDATES,
        "require_unique_candidates": True,
        "cache_implementation": "offloaded",
        "fixed_probe": {
            "path": str(probe_path),
            "rows": PROBE_ROWS,
            "sha256": _sha256(probe_path),
        },
        "trie_manifest_sha256": _sha256(trie_dir / "manifest.json"),
        "policy_adapter": {
            "path": str(policy_path),
            "adapter_config_sha256": _sha256(policy_path / "adapter_config.json"),
            "adapter_model_sha256": _sha256(policy_path / "adapter_model.safetensors"),
        },
        "runtime_signature_sha256": runtime_signature_sha256,
        "selected_indices": list(selected),
        "result_count": len(results),
        "pass_count": sum(bool(item["pass_at_64"]) for item in results),
        "elapsed_seconds": time.monotonic() - started,
        "peak_reserved_gib": (
            torch.cuda.max_memory_reserved(torch_device) / 1024**3
            if torch_device.type == "cuda"
            else 0.0
        ),
        "results": results,
    }
    _atomic_json(output, report)
    return report


__all__ = ["PROBE_ROWS", "load_fixed_probe", "run_fixed_probe64"]
