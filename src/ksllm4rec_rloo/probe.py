"""Fixed beam-16 constrained proxy evaluation for RLOO checkpoints."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from transformers import LogitsProcessorList

from ksllm4rec_grpo.constraint import (
    RecommendationGrammar,
    RecommendationLogitsProcessor,
)
from ksllm4rec_grpo.prompt import encode_prompt
from ksllm4rec_grpo.trie import SidPrefixTrie
from ksllm4rec_orpo.data import Sid

from . import contract
from .fingerprint import runtime_signature
from .integrity import file_record, require_file
from .modeling import load_policy_model


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
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
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _load_probe(path: Path) -> list[dict[str, Any]]:
    require_file(path, contract.FIXED_PROBE_SHA256)
    required = {
        "task",
        "system",
        "instruction",
        "target",
        "target_sid",
        "source_file",
        "source_line",
        "source_row_sha256",
    }
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            if not isinstance(value, dict) or set(value) != required:
                raise ValueError(f"Invalid fixed probe schema at line {line_number}.")
            rows.append(value)
    counts = Counter(row["task"] for row in rows)
    if counts != Counter({"text_to_sid": 512, "recommend": 512}):
        raise RuntimeError(f"Fixed probe task counts differ: {dict(counts)}")
    return rows


def probe_run_signature(
    config: dict[str, Any],
    policy_adapter_path: Path,
    trie_dir: Path,
    *,
    groups_path: Path,
    calibration_ids_path: Path,
    config_path: Path,
) -> dict[str, Any]:
    payload = {
        "training_runtime_signature": runtime_signature(
            config,
            groups_path,
            trie_dir,
            calibration_ids_path,
            config_path=config_path,
        ),
        "policy_adapter_weights": file_record(
            policy_adapter_path / "adapter_model.safetensors"
        ),
        "policy_adapter_config": file_record(
            policy_adapter_path / "adapter_config.json"
        ),
        "trie_manifest": require_file(
            trie_dir / "manifest.json", contract.TRIE_MANIFEST_SHA256
        ),
        "fixed_probe": require_file(
            Path(config["evaluation"]["fixed_probe"]),
            contract.FIXED_PROBE_SHA256,
        ),
        "num_beams": int(config["evaluation"]["num_beams"]),
        "max_completion_length": int(
            config["evaluation"]["max_completion_length"]
        ),
        "algorithm": {
            "constrained": True,
            "do_sample": False,
            "grammar": "frontier_sid_trie",
            "prompt": "qwen3_nothink",
        },
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "sha256": hashlib.sha256(canonical).hexdigest(),
        "inputs": payload,
    }


def _existing_predictions(path: Path, expected_signature: str) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_bytes()
    if raw and not raw.endswith(b"\n"):
        prefix, separator, _ = raw.rpartition(b"\n")
        repaired = prefix + separator if separator else b""
        temporary = path.with_name(f".{path.name}.repair")
        temporary.write_bytes(repaired)
        os.replace(temporary, path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            value = json.loads(line)
            if value.get("probe_index") != index:
                raise RuntimeError("Partial probe predictions have a broken cursor.")
            if value.get("run_signature") != expected_signature:
                raise RuntimeError("Partial probe predictions belong to another run.")
            rows.append(value)
    return rows


def run_fixed_probe(
    config: dict[str, Any],
    *,
    policy_adapter_path: Path,
    trie_dir: Path,
    groups_path: Path,
    calibration_ids_path: Path,
    config_path: Path,
    output_dir: Path,
    device: str = "cuda:0",
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    policy_adapter_path = Path(policy_adapter_path).resolve()
    trie_dir = Path(trie_dir).resolve()
    signature = probe_run_signature(
        config,
        policy_adapter_path,
        trie_dir,
        groups_path=groups_path,
        calibration_ids_path=calibration_ids_path,
        config_path=config_path,
    )
    state_path = output_dir / "probe_state.json"
    if state_path.exists():
        if json.loads(state_path.read_text(encoding="utf-8")) != signature:
            raise RuntimeError("Probe output directory belongs to different inputs.")
    else:
        if (output_dir / "predictions.jsonl").exists():
            raise RuntimeError("Probe predictions exist without an input signature.")
        _atomic_json(state_path, signature)

    report_path = output_dir / "probe_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
        completed = _existing_predictions(
            output_dir / "predictions.jsonl", signature["sha256"]
        )
        if report.get("run_signature") != signature or report.get("rows") != len(
            completed
        ):
            raise RuntimeError("Existing probe report failed input binding checks.")
        if len(completed) == 1024:
            return report

    rows = _load_probe(Path(config["evaluation"]["fixed_probe"]))
    trie = SidPrefixTrie.load(
        trie_dir, expected_leaf_count=contract.EXPECTED_TRIE_LEAVES
    )
    reachability = Counter()
    for row in rows:
        if trie.contains(Sid.parse(row["target_sid"])):
            reachability[row["task"]] += 1
    expected_reachability = {"text_to_sid": 512, "recommend": 512}
    if dict(reachability) != expected_reachability:
        raise RuntimeError(
            "Fixed probe reachability differs: "
            f"expected={expected_reachability}, actual={dict(reachability)}"
        )

    predictions_path = output_dir / "predictions.jsonl"
    completed = _existing_predictions(predictions_path, signature["sha256"])
    inference_config = copy.deepcopy(config)
    inference_config["train"]["gradient_checkpointing"] = False
    bundle = load_policy_model(
        inference_config,
        device=device,
        policy_adapter_path=policy_adapter_path,
    )
    text_grammar = RecommendationGrammar(
        bundle.tokenizer, trie, mode="probe_text_to_sid"
    )
    recommendation_grammar = RecommendationGrammar(
        bundle.tokenizer, trie, mode="probe_recommendation"
    )
    bundle.model.eval()
    started = time.monotonic()
    with predictions_path.open("a", encoding="utf-8") as handle:
        for index in range(len(completed), len(rows)):
            row = rows[index]
            grammar = (
                text_grammar
                if row["task"] == "text_to_sid"
                else recommendation_grammar
            )
            prompt_ids = encode_prompt(
                bundle.tokenizer,
                row["system"],
                row["instruction"],
                cutoff_len=int(config["data"]["cutoff_len"]),
            )
            input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
            attention_mask = torch.ones_like(input_ids)
            processor = RecommendationLogitsProcessor(grammar, len(prompt_ids))
            with torch.inference_mode():
                generated = bundle.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=int(
                        config["evaluation"]["max_completion_length"]
                    ),
                    do_sample=False,
                    num_beams=int(config["evaluation"]["num_beams"]),
                    renormalize_logits=True,
                    early_stopping=True,
                    use_cache=True,
                    pad_token_id=grammar.eos_token_id,
                    eos_token_id=grammar.eos_token_id,
                    logits_processor=LogitsProcessorList([processor]),
                )
            completion = generated[0, len(prompt_ids) :].tolist()
            predicted = grammar.parse(completion)
            target = Sid.parse(row["target_sid"])
            record = {
                "probe_index": index,
                "run_signature": signature["sha256"],
                "task": row["task"],
                "source_row_sha256": row["source_row_sha256"],
                "target_sid": target.render(),
                "target_reachable": trie.contains(target),
                "predicted_sid": predicted.render(),
                "valid_sid": trie.contains(predicted),
                "exact": predicted == target,
                "prompt_tokens": len(prompt_ids),
                "completion_tokens": len(completion),
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            if (index + 1) % 16 == 0:
                handle.flush()
                os.fsync(handle.fileno())
        handle.flush()
        os.fsync(handle.fileno())

    predictions = _existing_predictions(predictions_path, signature["sha256"])
    metrics: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "rows": 0,
            "reachable_targets": 0,
            "valid_predictions": 0,
            "exact": 0,
            "exact_reachable": 0,
        }
    )
    for row in predictions:
        bucket = metrics[row["task"]]
        bucket["rows"] += 1
        bucket["reachable_targets"] += int(row["target_reachable"])
        bucket["valid_predictions"] += int(row["valid_sid"])
        bucket["exact"] += int(row["exact"])
        bucket["exact_reachable"] += int(
            row["exact"] and row["target_reachable"]
        )
    report = {
        "schema_version": 1,
        "run_signature": signature,
        "policy_adapter": str(policy_adapter_path),
        "beam_size": int(config["evaluation"]["num_beams"]),
        "legal_sid_universe": config["trie"]["strategy"],
        "rows": len(predictions),
        "seconds_this_invocation": time.monotonic() - started,
        "metrics": dict(metrics),
    }
    _atomic_json(report_path, report)
    return report


__all__ = ["probe_run_signature", "run_fixed_probe"]
