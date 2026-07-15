"""Tokenization audit and exact-length synthetic memory-gate pair."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer

from ksllm4rec_sft.data import SourceRecord, render_qwen3_nothink, sha256_file
from ksllm4rec_sft.manifest import atomic_write_json

from .contract import GATE_CUTOFFS, GATE_DATASET_NAME
from .data import EMPTY_THINK, EXPECTED_RECORDS, iter_pair_rows
from .miner import audit_final_pairs
from .pairs import _atomic_jsonl


def _infer_seqlen(source_len: int, target_len: int, cutoff_len: int) -> tuple[int, int]:
    """Exact copy of the pinned LLaMA-Factory pair truncation arithmetic."""

    if target_len * 2 < cutoff_len:
        max_target_len = cutoff_len
    elif source_len * 2 < cutoff_len:
        max_target_len = cutoff_len - source_len
    else:
        max_target_len = int(cutoff_len * (target_len / (source_len + target_len)))
    new_target_len = min(max_target_len, target_len)
    max_source_len = max(cutoff_len - new_target_len, 0)
    new_source_len = min(max_source_len, source_len)
    return new_source_len, new_target_len


def _token_lengths(tokenizer, row: dict[str, str]) -> tuple[int, int, int, int]:
    source, chosen_target = render_qwen3_nothink(
        SourceRecord(row["system"], row["instruction"], row["chosen"])
    )
    _, rejected_target = render_qwen3_nothink(
        SourceRecord(row["system"], row["instruction"], row["rejected"])
    )
    source_len = len(tokenizer.encode(source, add_special_tokens=False))
    chosen_len = len(tokenizer.encode(chosen_target, add_special_tokens=False))
    rejected_len = len(tokenizer.encode(rejected_target, add_special_tokens=False))
    return source_len, chosen_len, rejected_len, source_len + max(chosen_len, rejected_len)


def _write_gate_pair(tokenizer, data_dir: Path) -> dict[str, Any]:
    chosen = EMPTY_THINK + "\n<|video_begin|><s_a_1><s_b_1><s_c_1>"
    rejected = EMPTY_THINK + "\n<|video_begin|><s_a_1><s_b_1><s_c_2>"
    instruction = "显存门禁序列：" + ("测试 " * 20_000) + "/no_think"
    row = {
        "instruction": instruction,
        "input": "",
        "chosen": chosen,
        "rejected": rejected,
        "system": "",
    }
    source_len, chosen_len, rejected_len, raw_max = _token_lengths(tokenizer, row)
    if raw_max <= 16_384:
        raise RuntimeError(f"Synthetic gate row is too short: {raw_max}")
    processed_lengths: dict[str, int] = {}
    for stage, cutoff in GATE_CUTOFFS.items():
        source_after, target_after = _infer_seqlen(
            source_len, max(chosen_len, rejected_len), cutoff
        )
        processed = source_after + target_after
        if processed != cutoff or target_after < 1:
            raise RuntimeError(
                f"Gate {stage} does not materialize cutoff={cutoff}: got {processed}."
            )
        processed_lengths[stage] = processed
    path = data_dir / "memory_gate.jsonl"
    _atomic_jsonl(path, [row])
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": sha256_file(path),
        "source_tokens": source_len,
        "chosen_tokens": chosen_len,
        "rejected_tokens": rejected_len,
        "raw_max_tokens": raw_max,
        "processed_lengths": processed_lengths,
    }


def run_preflight(
    model_path: Path,
    data_dir: Path,
    report_path: Path,
    *,
    cutoff_len: int = 16_384,
) -> dict[str, Any]:
    model_path = model_path.resolve()
    data_dir = data_dir.resolve()
    train_path = data_dir / "train.jsonl"
    audit_path = data_dir / "audit.jsonl"
    dataset_info_path = data_dir / "dataset_info.json"
    audit = audit_final_pairs(audit_path, train_path)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )

    count = 0
    max_source = 0
    max_chosen = 0
    max_rejected = 0
    max_raw_pair = 0
    truncated = Counter()
    processed_min = cutoff_len
    processed_max = 0
    for row in iter_pair_rows(train_path):
        source_len, chosen_len, rejected_len, raw_pair = _token_lengths(tokenizer, row)
        count += 1
        max_source = max(max_source, source_len)
        max_chosen = max(max_chosen, chosen_len)
        max_rejected = max(max_rejected, rejected_len)
        max_raw_pair = max(max_raw_pair, raw_pair)
        target_len = max(chosen_len, rejected_len)
        source_after, target_after = _infer_seqlen(source_len, target_len, cutoff_len)
        processed = source_after + target_after
        processed_min = min(processed_min, processed)
        processed_max = max(processed_max, processed)
        truncated["rows"] += raw_pair > cutoff_len
        truncated["source"] += source_after < source_len
        truncated["target"] += target_after < target_len
        if target_after < 1:
            raise RuntimeError(f"Pair row {count} loses every response token.")
    if count != EXPECTED_RECORDS:
        raise RuntimeError(f"Preflight expected {EXPECTED_RECORDS} pairs, found {count}.")

    gate = _write_gate_pair(tokenizer, data_dir)
    dataset_info = json.loads(dataset_info_path.read_text(encoding="utf-8"))
    if GATE_DATASET_NAME not in dataset_info:
        raise RuntimeError("dataset_info.json does not declare the frozen gate dataset.")
    report = {
        "schema_version": 1,
        "status": "passed",
        "model_path": str(model_path),
        "train_path": str(train_path),
        "train_sha256": sha256_file(train_path),
        "audit": audit,
        "cutoff_len": cutoff_len,
        "records": count,
        "token_lengths": {
            "max_source": max_source,
            "max_chosen_response": max_chosen,
            "max_rejected_response": max_rejected,
            "max_raw_pair": max_raw_pair,
            "processed_min": processed_min,
            "processed_max": processed_max,
        },
        "truncated": dict(truncated),
        "memory_gate": gate,
        "dataset_info_sha256": sha256_file(dataset_info_path),
    }
    atomic_write_json(report_path, report)
    return report
