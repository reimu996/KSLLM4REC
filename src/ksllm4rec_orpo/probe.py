"""Prepare and evaluate the fixed Explorer generalization probe."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from ksllm4rec_sft.data import SourceRecord, iter_source_records, render_qwen3_nothink, sha256_file
from ksllm4rec_sft.manifest import atomic_write_json

from .data import direct_response, final_answer_suffix, find_sids, normalize_prompt_mode, source_row_sha256
from .pairs import _atomic_jsonl, _numeric_path_key


PROBE_PER_TASK = 512


def _explorer_records(directory: Path) -> Iterator[tuple[Path, int, SourceRecord]]:
    for path in sorted(directory.glob("*.jsonl"), key=_numeric_path_key):
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = json.loads(line)
                if not isinstance(value, list) or len(value) != 1:
                    raise ValueError(f"Invalid Explorer probe row at {path}:{line_number}")
                row = value[0]
                yield path, line_number, SourceRecord(
                    row["system"], row["prompt"], row["response"]
                )


def prepare_fixed_probe(
    baseline_path: Path,
    text_to_sid_dir: Path,
    recommend_dir: Path,
    output_path: Path,
    report_path: Path,
) -> dict[str, Any]:
    baseline_hashes = {source_row_sha256(record) for record in iter_source_records(baseline_path)}
    rows: list[dict[str, Any]] = []
    for task, directory in (
        ("text_to_sid", text_to_sid_dir),
        ("recommend", recommend_dir),
    ):
        accepted = 0
        for path, line_number, record in _explorer_records(directory):
            source_hash = source_row_sha256(record)
            if source_hash in baseline_hashes:
                continue
            normalized = direct_response(record.response)
            target_sids = find_sids(final_answer_suffix(normalized))
            if len(target_sids) != 1:
                continue
            rows.append(
                {
                    "task": task,
                    "system": record.system,
                    "instruction": normalize_prompt_mode(record.prompt),
                    "target": normalized,
                    "target_sid": target_sids[0].render(),
                    "source_file": str(path.resolve()),
                    "source_line": line_number,
                    "source_row_sha256": source_hash,
                }
            )
            accepted += 1
            if accepted == PROBE_PER_TASK:
                break
        if accepted != PROBE_PER_TASK:
            raise RuntimeError(f"Probe task {task} produced {accepted}, need {PROBE_PER_TASK}.")
    _atomic_jsonl(output_path, rows)
    counts = Counter(row["task"] for row in rows)
    report = {
        "schema_version": 1,
        "status": "passed",
        "records": len(rows),
        "task_counts": dict(sorted(counts.items())),
        "excluded_exact_baseline_rows": True,
        "probe_path": str(output_path.resolve()),
        "probe_size": output_path.stat().st_size,
        "probe_sha256": sha256_file(output_path),
    }
    atomic_write_json(report_path, report)
    return report


def _iter_probe(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            yield json.loads(line)


def evaluate_probe(
    model_path: Path,
    probe_path: Path,
    output_dir: Path,
    *,
    adapter_path: Path | None,
    batch_size: int = 4,
    min_free_gib: float = 20.5,
) -> dict[str, Any]:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for fixed-probe evaluation.")
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    free_gib = free_bytes / 1024**3
    if free_gib < min_free_gib:
        raise RuntimeError(
            f"Probe GPU gate failed: {free_gib:.3f} GiB free, need {min_free_gib:.3f}."
        )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    tokenizer.padding_side = "left"
    base = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).to("cuda:0")
    model = (
        PeftModel.from_pretrained(base, adapter_path, is_trainable=False)
        if adapter_path is not None
        else base
    )
    model.eval()
    rows = list(_iter_probe(probe_path))
    predictions: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for start in range(0, len(rows), batch_size):
        chunk = rows[start : start + batch_size]
        prompts = [
            render_qwen3_nothink(
                SourceRecord(row["system"], row["instruction"], row["target"])
            )[0]
            for row in chunk
        ]
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=16_320,
            add_special_tokens=False,
        )
        inputs = {key: value.to("cuda:0") for key, value in inputs.items()}
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                max_new_tokens=64,
                use_cache=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        prompt_width = inputs["input_ids"].size(1)
        for row, token_ids in zip(chunk, generated[:, prompt_width:]):
            text = tokenizer.decode(token_ids, skip_special_tokens=False)
            sids = find_sids(text)
            predicted_sid = sids[0].render() if sids else None
            correct = predicted_sid == row["target_sid"]
            counts[f"{row['task']}_total"] += 1
            counts[f"{row['task']}_valid_sid"] += predicted_sid is not None
            counts[f"{row['task']}_exact_sid"] += correct
            predictions.append(
                {
                    "task": row["task"],
                    "target_sid": row["target_sid"],
                    "predicted_sid": predicted_sid,
                    "exact_sid": correct,
                    "generated_text": text,
                    "source_row_sha256": row["source_row_sha256"],
                }
            )
    output_dir.mkdir(parents=True, exist_ok=False)
    prediction_path = output_dir / "predictions.jsonl"
    _atomic_jsonl(prediction_path, predictions)
    metrics: dict[str, Any] = {}
    for task in ("text_to_sid", "recommend"):
        total = counts[f"{task}_total"]
        metrics[task] = {
            "records": total,
            "valid_sid_rate": counts[f"{task}_valid_sid"] / total,
            "exact_sid_rate": counts[f"{task}_exact_sid"] / total,
        }
    report = {
        "status": "passed",
        "model_path": str(model_path.resolve()),
        "adapter_path": str(adapter_path.resolve()) if adapter_path else None,
        "probe_sha256": sha256_file(probe_path),
        "metrics": metrics,
        "predictions_sha256": sha256_file(prediction_path),
        "gpu_free_gib_before": free_gib,
        "gpu_total_gib": total_bytes / 1024**3,
        "peak_memory_reserved_gib": torch.cuda.max_memory_reserved() / 1024**3,
    }
    atomic_write_json(output_dir / "report.json", report)
    return report
