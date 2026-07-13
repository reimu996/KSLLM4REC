"""CPU and tokenizer preflight checks for the approved SFT run."""

from __future__ import annotations

import math
from collections import Counter
from pathlib import Path

import llamafactory
from transformers import AutoTokenizer

from llamafactory.data import Role, TEMPLATES

from .data import iter_derived_records, render_qwen3_nothink
from .integrity import verify_artifact_lock
from .item_tokens import build_item_token_ids
from .manifest import atomic_write_json


def _percentile(sorted_values: list[int], quantile: float) -> int:
    if not sorted_values:
        raise ValueError("Cannot compute a percentile of an empty list.")
    index = math.ceil(quantile * len(sorted_values)) - 1
    return sorted_values[max(0, min(index, len(sorted_values) - 1))]


def tokenizer_preflight(
    model_path: Path,
    derived_path: Path,
    report_path: Path,
    *,
    cutoff_len: int = 32768,
    artifact_lock: Path | None = None,
) -> dict:
    integrity = (
        verify_artifact_lock(
            artifact_lock,
            llamafactory_module_file=Path(llamafactory.__file__),
        )
        if artifact_lock is not None
        else None
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True, local_files_only=True
    )
    item_token_ids = build_item_token_ids(tokenizer)
    item_token_id_set = set(item_token_ids)
    template = TEMPLATES["qwen3_nothink"]
    if type(template).__name__ != "Template":
        raise RuntimeError(
            f"qwen3_nothink unexpectedly uses {type(template).__name__}."
        )
    template.fix_special_tokens(tokenizer)
    if tokenizer.eos_token != "<|im_end|>":
        raise RuntimeError(
            f"Unexpected qwen3_nothink EOS token: {tokenizer.eos_token!r}"
        )

    lengths: list[int] = []
    source_lengths: list[int] = []
    target_lengths: list[int] = []
    counters: Counter[str] = Counter()
    longest: list[tuple[int, int, int, int]] = []
    render_check_lines: list[int] = []
    think_token_id = tokenizer.convert_tokens_to_ids("<think>")

    for index, record in enumerate(iter_derived_records(derived_path)):
        messages = [
            {"role": Role.USER, "content": record.prompt},
            {"role": Role.ASSISTANT, "content": record.response},
        ]
        source_ids, target_ids = template.encode_oneturn(
            tokenizer, messages, record.system
        )
        if index % 4096 == 0:
            source, target = render_qwen3_nothink(record)
            if source_ids != tokenizer.encode(source, add_special_tokens=False):
                raise RuntimeError(
                    f"Template source rendering mismatch at line {index + 1}."
                )
            if target_ids != tokenizer.encode(target, add_special_tokens=False):
                raise RuntimeError(
                    f"Template target rendering mismatch at line {index + 1}."
                )
            render_check_lines.append(index + 1)

        source_len = len(source_ids)
        target_len = len(target_ids)
        total_len = source_len + target_len
        counters["target_tokens"] += target_len
        counters["target_item_tokens"] += sum(
            token_id in item_token_id_set for token_id in target_ids
        )
        counters["responses_with_think_text"] += "<think>" in record.response
        counters["targets_with_think_token"] += think_token_id in target_ids
        if target_len > cutoff_len:
            counters["target_over_cutoff"] += 1
        if total_len > cutoff_len:
            counters["total_over_cutoff"] += 1
        lengths.append(total_len)
        source_lengths.append(source_len)
        target_lengths.append(target_len)
        longest.append((total_len, index + 1, source_len, target_len))

    lengths.sort()
    source_lengths.sort()
    target_lengths.sort()
    longest.sort(reverse=True)
    report = {
        "model_path": str(model_path.resolve()),
        "derived_path": str(derived_path.resolve()),
        "records": len(lengths),
        "tokenizer_vocab_size": len(tokenizer),
        "item_token_count": len(item_token_ids),
        "item_token_id_range": [item_token_ids[0], item_token_ids[-1]],
        "integrity": integrity,
        "template": {
            "name": "qwen3_nothink",
            "class": type(template).__name__,
            "eos_token": tokenizer.eos_token,
            "render_equivalence_checked_lines": render_check_lines,
        },
        "cutoff_len": cutoff_len,
        "total_over_cutoff": counters["total_over_cutoff"],
        "target_over_cutoff": counters["target_over_cutoff"],
        "lengths": {
            "min": lengths[0],
            "p50": _percentile(lengths, 0.50),
            "p90": _percentile(lengths, 0.90),
            "p95": _percentile(lengths, 0.95),
            "p99": _percentile(lengths, 0.99),
            "max": lengths[-1],
        },
        "source_max": source_lengths[-1],
        "target_max": target_lengths[-1],
        "targets": {
            "tokens": counters["target_tokens"],
            "item_tokens": counters["target_item_tokens"],
            "item_ratio": counters["target_item_tokens"] / counters["target_tokens"],
            "responses_with_think_text": counters["responses_with_think_text"],
            "targets_with_think_token": counters["targets_with_think_token"],
        },
        "longest_records": [
            {"line": line, "total": total, "source": source, "target": target}
            for total, line, source, target in longest[:20]
        ],
    }
    blockers = []
    if counters["target_over_cutoff"]:
        blockers.append(
            "At least one response exceeds cutoff_len; response truncation is forbidden."
        )
    if counters["total_over_cutoff"]:
        blockers.append(
            "At least one full sample exceeds cutoff_len; explicit prompt left-cropping is required before training."
        )
    report["status"] = "failed" if blockers else "passed"
    report["blockers"] = blockers
    atomic_write_json(report_path, report)
    if blockers:
        raise RuntimeError(" ".join(blockers))
    return report
