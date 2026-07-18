"""Immutable conversion and validation for the competition SFT JSONL."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

from .profiles import BASELINE_PROFILE

DATASET_NAME = BASELINE_PROFILE.dataset_name
REQUIRED_FIELDS = ("system", "prompt", "response")


@dataclass(frozen=True)
class SourceRecord:
    system: str
    prompt: str
    response: str


@dataclass(frozen=True)
class DataReport:
    source_path: str
    source_sha256: str
    source_bytes: int
    records: int
    content_sha256: str
    derived_path: str
    derived_sha256: str
    derived_content_sha256: str
    responses_with_think: int
    responses_with_empty_think: int
    prompts_with_no_think: int
    max_prompt_chars: int
    max_response_chars: int


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _update_content_digest(digest: "hashlib._Hash", record: SourceRecord) -> None:
    for field in REQUIRED_FIELDS:
        payload = getattr(record, field).encode("utf-8")
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)


def iter_source_records(path: Path) -> Iterator[SourceRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {line_number}: {exc}") from exc
            if (
                not isinstance(value, list)
                or len(value) != 1
                or not isinstance(value[0], dict)
            ):
                raise ValueError(
                    f"Line {line_number} must be a one-element JSON array containing an object."
                )
            row = value[0]
            for field in REQUIRED_FIELDS:
                if field not in row or not isinstance(row[field], str):
                    raise ValueError(
                        f"Line {line_number} field {field!r} must be a string."
                    )
            if not row["response"]:
                raise ValueError(f"Line {line_number} has an empty response.")
            yield SourceRecord(row["system"], row["prompt"], row["response"])


def iter_derived_records(path: Path) -> Iterator[SourceRecord]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            row = json.loads(line)
            expected = {"instruction", "input", "output", "system"}
            if not isinstance(row, dict) or set(row) != expected or row["input"] != "":
                raise ValueError(f"Invalid derived record at line {line_number}.")
            yield SourceRecord(row["system"], row["instruction"], row["output"])


def _atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


def expected_dataset_info(dataset_name: str, derived_filename: str) -> dict:
    if not dataset_name:
        raise ValueError("dataset_name must not be empty.")
    if not derived_filename:
        raise ValueError("derived_filename must not be empty.")
    return {
        dataset_name: {
            "file_name": derived_filename,
            "formatting": "alpaca",
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "system": "system",
            },
        }
    }


def validate_dataset_info(
    path: Path, *, dataset_name: str, derived_filename: str
) -> dict:
    try:
        actual = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid dataset_info.json at {path}: {exc}") from exc
    expected = expected_dataset_info(dataset_name, derived_filename)
    if actual != expected:
        raise RuntimeError(
            f"dataset_info.json does not exactly match dataset {dataset_name!r}: "
            f"expected {expected!r}, got {actual!r}"
        )
    return actual


def prepare_dataset(
    source_path: Path,
    output_dir: Path,
    *,
    dataset_name: str = DATASET_NAME,
) -> DataReport:
    if not dataset_name:
        raise ValueError("dataset_name must not be empty.")
    source_path = source_path.resolve()
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    derived_path = output_dir / "train_alpaca.jsonl"

    source_digest = hashlib.sha256()
    counters: Counter[str] = Counter()
    max_prompt_chars = 0
    max_response_chars = 0
    fd, temp_name = tempfile.mkstemp(prefix=".train_alpaca.", dir=output_dir)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as target:
            for record in iter_source_records(source_path):
                _update_content_digest(source_digest, record)
                counters["records"] += 1
                counters["responses_with_think"] += "<think>" in record.response
                counters["responses_with_empty_think"] += (
                    "<think></think>" in record.response
                    or "<think>\n</think>" in record.response
                )
                counters["prompts_with_no_think"] += "/no_think" in record.prompt
                max_prompt_chars = max(max_prompt_chars, len(record.prompt))
                max_response_chars = max(max_response_chars, len(record.response))
                converted = {
                    "instruction": record.prompt,
                    "input": "",
                    "output": record.response,
                    "system": record.system,
                }
                target.write(
                    json.dumps(converted, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
            target.flush()
            os.fsync(target.fileno())
        os.replace(temp_name, derived_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)

    derived_digest = hashlib.sha256()
    derived_count = 0
    for record in iter_derived_records(derived_path):
        _update_content_digest(derived_digest, record)
        derived_count += 1

    if (
        derived_count != counters["records"]
        or derived_digest.hexdigest() != source_digest.hexdigest()
    ):
        raise RuntimeError(
            "Derived data did not preserve source record content exactly."
        )

    dataset_info = expected_dataset_info(dataset_name, derived_path.name)
    _atomic_json(output_dir / "dataset_info.json", dataset_info)

    report = DataReport(
        source_path=str(source_path),
        source_sha256=sha256_file(source_path),
        source_bytes=source_path.stat().st_size,
        records=counters["records"],
        content_sha256=source_digest.hexdigest(),
        derived_path=str(derived_path),
        derived_sha256=sha256_file(derived_path),
        derived_content_sha256=derived_digest.hexdigest(),
        responses_with_think=counters["responses_with_think"],
        responses_with_empty_think=counters["responses_with_empty_think"],
        prompts_with_no_think=counters["prompts_with_no_think"],
        max_prompt_chars=max_prompt_chars,
        max_response_chars=max_response_chars,
    )
    _atomic_json(output_dir / "data_report.json", asdict(report))
    return report


def render_qwen3_nothink(record: SourceRecord) -> tuple[str, str]:
    """Render exactly the LLaMA-Factory qwen3_nothink single-turn layout."""

    system = f"<|im_start|>system\n{record.system}<|im_end|>\n" if record.system else ""
    source = (
        f"{system}<|im_start|>user\n{record.prompt}<|im_end|>\n<|im_start|>assistant\n"
    )
    target = f"{record.response}<|im_end|>\n"
    return source, target
