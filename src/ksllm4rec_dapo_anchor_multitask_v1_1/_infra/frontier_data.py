"""Vendored Frontier source/provenance reader used by Multitask V1."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator

from .sft_data import SourceRecord


FRONTIER_TASKS = (
    "recommendation",
    "item_text_to_sid",
    "item_sid_to_text",
    "world_choice",
    "user_selection",
    "user_evolution",
    "item_multilevel_listwise_to_sid",
)


@dataclass(frozen=True)
class FrontierIndexedRecord:
    """One source row joined byte-for-byte to its provenance row."""

    line_number: int
    source_sha256: str
    record: SourceRecord
    task: str
    variant: str | None
    route_id: str
    provenance: dict[str, Any]


def _raw_line_hash(raw: bytes) -> str:
    return hashlib.sha256(raw[:-1] if raw.endswith(b"\n") else raw).hexdigest()


def _parse_source(raw: bytes, line_number: int) -> SourceRecord:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid Frontier source JSON at line {line_number}.") from exc
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], dict):
        raise ValueError(
            f"Frontier source line {line_number} must be a one-element object list."
        )
    row = value[0]
    fields = ("system", "prompt", "response")
    if any(field not in row or not isinstance(row[field], str) for field in fields):
        raise ValueError(f"Frontier source line {line_number} has invalid fields.")
    if not row["response"]:
        raise ValueError(f"Frontier source line {line_number} has an empty response.")
    return SourceRecord(row["system"], row["prompt"], row["response"])


def iter_frontier_records(
    source: Path, provenance: Path
) -> Iterator[FrontierIndexedRecord]:
    """Yield aligned rows and reject any source/provenance drift."""

    source_path = Path(source)
    provenance_path = Path(provenance)
    sentinel = object()
    with source_path.open("rb") as source_handle, provenance_path.open(
        "r", encoding="utf-8"
    ) as provenance_handle:
        for line_number, pair in enumerate(
            zip_longest(source_handle, provenance_handle, fillvalue=sentinel), start=1
        ):
            raw, provenance_line = pair
            if raw is sentinel or provenance_line is sentinel:
                raise ValueError("Frontier train/provenance line counts differ.")
            source_sha256 = _raw_line_hash(raw)
            record = _parse_source(raw, line_number)
            try:
                metadata = json.loads(provenance_line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid Frontier provenance JSON at line {line_number}."
                ) from exc
            if not isinstance(metadata, dict):
                raise ValueError(
                    f"Frontier provenance line {line_number} is not an object."
                )
            if metadata.get("candidate_line") != line_number:
                raise ValueError(
                    f"Frontier provenance line mismatch at {line_number}: "
                    f"candidate_line={metadata.get('candidate_line')!r}."
                )
            if metadata.get("record_sha256") != source_sha256:
                raise ValueError(
                    f"Frontier source/provenance hash mismatch at line {line_number}."
                )
            task = metadata.get("task")
            if task not in FRONTIER_TASKS:
                raise ValueError(
                    f"Unknown Frontier task at line {line_number}: {task!r}."
                )
            nested = metadata.get("source")
            variant = nested.get("variant") if isinstance(nested, dict) else None
            route_id = metadata.get("route_id")
            if not isinstance(route_id, str) or not route_id:
                raise ValueError(f"Missing route_id at Frontier line {line_number}.")
            yield FrontierIndexedRecord(
                line_number=line_number,
                source_sha256=source_sha256,
                record=record,
                task=task,
                variant=variant if isinstance(variant, str) else None,
                route_id=route_id,
                provenance=metadata,
            )


__all__ = ["FRONTIER_TASKS", "FrontierIndexedRecord", "iter_frontier_records"]
