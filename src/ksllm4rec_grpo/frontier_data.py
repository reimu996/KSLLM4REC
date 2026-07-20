"""Deterministic data artifacts for the Frontier-started GRPO profile.

This module deliberately does not use the frozen baseline contracts in
``ksllm4rec_grpo.data`` or ``contract.py``.  The Frontier export has its own
provenance file and its own task names, counts, groups, and SID universe.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter
from dataclasses import dataclass
from itertools import zip_longest
from pathlib import Path
from typing import Any, Iterator

from ksllm4rec_orpo.data import (
    Sid,
    direct_response,
    find_sids,
    final_answer_suffix,
    normalize_prompt_mode,
)
from ksllm4rec_sft.data import SourceRecord

from .trie import SidPrefixTrie


FRONTIER_SOURCE_RECORDS = 63_700
FRONTIER_RECOMMEND_ROWS = 30_902
FRONTIER_RECOMMEND_GROUPS = 17_016
FRONTIER_RECOMMEND_POSITIVE_EDGES = 30_465
FRONTIER_RECOMMEND_UNIQUE_SIDS = 29_414
FRONTIER_RECOMMEND_DUPLICATE_EDGES = 437
FRONTIER_RECOMMEND_DUPLICATE_GROUPS = 129
FRONTIER_TRIE_LEAVES = 905_469
FRONTIER_TRIE_A_NODES = 10_744
FRONTIER_TRIE_AB_NODES = 429_540
FRONTIER_PROBE_ROWS_PER_TASK = 512

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
    """One source row joined to its provenance row.

    ``source_sha256`` is the provenance hash: SHA256 of the raw JSONL line
    after removing only the trailing LF.  It is intentionally distinct from
    the canonical three-field hash used by the old baseline loader.
    """

    line_number: int
    source_sha256: str
    record: SourceRecord
    task: str
    variant: str | None
    route_id: str
    provenance: dict[str, Any]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_record(path: Path) -> dict[str, Any]:
    path = Path(path).resolve()
    return {"path": str(path), "size": path.stat().st_size, "sha256": sha256_file(path)}


def _json_dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _canonical_group_id(system: str, prompt: str) -> str:
    payload = json.dumps([system, prompt], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _raw_line_hash(raw: bytes) -> str:
    # The Frontier provenance producer hashes the line without LF, but keeps
    # every other byte unchanged (including a possible CR).
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


def iter_frontier_records(source: Path, provenance: Path) -> Iterator[FrontierIndexedRecord]:
    """Yield source/provenance pairs while enforcing byte-level alignment."""

    source = Path(source)
    provenance = Path(provenance)
    sentinel = object()
    with source.open("rb") as source_handle, provenance.open(
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
                raise ValueError(f"Frontier provenance line {line_number} is not an object.")
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
                raise ValueError(f"Unknown Frontier task at line {line_number}: {task!r}.")
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


def build_source_lock_report(source: Path, provenance: Path) -> dict[str, Any]:
    """Scan both files and return a reproducible, lightweight audit report."""

    task_counts: Counter[str] = Counter()
    variant_counts: Counter[str] = Counter()
    route_counts: Counter[str] = Counter()
    records = 0
    for item in iter_frontier_records(source, provenance):
        records += 1
        task_counts[item.task] += 1
        variant_counts[item.variant or "(none)"] += 1
        route_counts[f"{item.route_id}:{item.task}"] += 1
    if records != FRONTIER_SOURCE_RECORDS:
        raise RuntimeError(
            f"Frontier source row count mismatch: expected {FRONTIER_SOURCE_RECORDS}, "
            f"got {records}."
        )
    return {
        "schema_version": 1,
        "kind": "frontier_source_lock",
        "source": _file_record(Path(source)),
        "provenance": _file_record(Path(provenance)),
        "records": records,
        "task_counts": dict(sorted(task_counts.items())),
        "variant_counts": dict(sorted(variant_counts.items())),
        "route_task_counts": dict(sorted(route_counts.items())),
        "alignment": "candidate_line_and_raw_line_sha256_without_lf",
    }


def _recommendation_parts(item: FrontierIndexedRecord) -> tuple[str, str]:
    prompt = normalize_prompt_mode(item.record.prompt)
    suffix = final_answer_suffix(item.record.response)
    sids = find_sids(suffix)
    if len(sids) != 1:
        raise ValueError(
            f"Frontier recommendation line {item.line_number} must contain exactly "
            f"one final SID, got {len(sids)}."
        )
    return prompt, sids[0].render()


def _write_groups_file(path: Path, groups: list[dict[str, Any]]) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for row in groups:
            raw = (
                json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
            ).encode("utf-8")
            handle.write(raw)
            digest.update(raw)
    return {
        "path": str(path.resolve()),
        "rows": len(groups),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def build_frontier_groups(
    source: Path,
    provenance: Path,
    output_dir: Path,
    *,
    enforce_contract: bool = True,
) -> dict[str, Any]:
    """Build normalized recommendation groups from Frontier provenance labels."""

    source = Path(source).resolve()
    provenance = Path(provenance).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Frontier groups directory: {output_dir}"
        )

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    task_counts: Counter[str] = Counter()
    source_records = 0
    raw_recommendation_rows = 0
    raw_positive_edges = 0
    for item in iter_frontier_records(source, provenance):
        source_records += 1
        task_counts[item.task] += 1
        if item.task != "recommendation":
            continue
        raw_recommendation_rows += 1
        prompt, sid = _recommendation_parts(item)
        key = (item.record.system, prompt)
        state = grouped.setdefault(
            key,
            {"positive_sids": set(), "source_lines": [], "variants": Counter()},
        )
        state["positive_sids"].add(sid)
        state["source_lines"].append(item.line_number)
        state["variants"][item.variant or "(none)"] += 1
        raw_positive_edges += 1

    rows: list[dict[str, Any]] = []
    for (system, prompt), state in grouped.items():
        rows.append(
            {
                "schema_version": 1,
                "group_id": _canonical_group_id(system, prompt),
                "system": system,
                "prompt": prompt,
                "positive_sids": sorted(state["positive_sids"]),
                "source_lines": sorted(state["source_lines"]),
            }
        )
    rows.sort(key=lambda row: row["group_id"])

    positive_edges = sum(len(row["positive_sids"]) for row in rows)
    duplicate_edges = raw_positive_edges - positive_edges
    duplicate_groups = sum(
        1
        for row in rows
        if len(row["source_lines"]) > len(row["positive_sids"])
    )
    unique_positive_sids = len(
        {sid for row in rows for sid in row["positive_sids"]}
    )
    if enforce_contract:
        expected = {
            "raw_recommendation_rows": FRONTIER_RECOMMEND_ROWS,
            "groups": FRONTIER_RECOMMEND_GROUPS,
            "positive_edges": FRONTIER_RECOMMEND_POSITIVE_EDGES,
            "unique_positive_sids": FRONTIER_RECOMMEND_UNIQUE_SIDS,
            "duplicate_edges": FRONTIER_RECOMMEND_DUPLICATE_EDGES,
            "duplicate_groups": FRONTIER_RECOMMEND_DUPLICATE_GROUPS,
        }
        actual = {
            "raw_recommendation_rows": raw_recommendation_rows,
            "groups": len(rows),
            "positive_edges": positive_edges,
            "unique_positive_sids": unique_positive_sids,
            "duplicate_edges": duplicate_edges,
            "duplicate_groups": duplicate_groups,
        }
        if actual != expected:
            raise RuntimeError(f"Frontier recommendation contract mismatch: {actual}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
    )
    try:
        groups_record = _write_groups_file(temporary / "groups.jsonl", rows)
        # The file is written in a temporary directory for atomic publication;
        # manifests must bind the final stable path, not the staging path.
        groups_record["path"] = str((output_dir / "groups.jsonl").resolve())
        source_lock = {
            "source": _file_record(source),
            "provenance": _file_record(provenance),
            "records": source_records,
            "task_counts": dict(sorted(task_counts.items())),
        }
        manifest = {
            "schema_version": 4,
            "spec_version": "2.0",
            "kind": "frontier_recommendation_prompt_groups_without_completions",
            "source_lock": source_lock,
            "normalization": {
                "prompt_mode": "/think|/no_think -> /no_think",
                "reward_target": "final SID after </think>; thinking text ignored",
                "duplicate_policy": "raw rows retained in source_lines; positive_sids set",
            },
            "groups": groups_record,
            "raw_recommendation_rows": raw_recommendation_rows,
            "positive_rows": positive_edges,
            "raw_positive_rows": raw_positive_edges,
            "duplicate_edges": duplicate_edges,
            "duplicate_groups": duplicate_groups,
            "unique_positive_sids": unique_positive_sids,
            "positive_set_size_distribution": dict(
                sorted(Counter(len(row["positive_sids"]) for row in rows).items())
            ),
            "completion_fields": [],
        }
        _json_dump(temporary / "data_manifest.json", manifest)
        os.replace(temporary, output_dir)
        return manifest
    except BaseException:
        import shutil

        shutil.rmtree(temporary, ignore_errors=True)
        raise


def build_frontier_trie(
    source: Path,
    provenance: Path,
    output_dir: Path,
    *,
    enforce_contract: bool = True,
) -> dict[str, Any]:
    """Build the legal SID trie from every system/prompt/response field."""

    source = Path(source).resolve()
    provenance = Path(provenance).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite existing Frontier trie directory: {output_dir}"
        )
    sids: set[Sid] = set()
    task_counts: Counter[str] = Counter()
    records = 0
    for item in iter_frontier_records(source, provenance):
        records += 1
        task_counts[item.task] += 1
        for field in (item.record.system, item.record.prompt, item.record.response):
            sids.update(find_sids(field))
    trie = SidPrefixTrie.from_sids(sids)
    if enforce_contract:
        counts = trie.counts
        expected = {
            "leaves": FRONTIER_TRIE_LEAVES,
            "a_nodes": FRONTIER_TRIE_A_NODES,
            "ab_nodes": FRONTIER_TRIE_AB_NODES,
        }
        actual = {key: counts[key] for key in expected}
        if actual != expected:
            raise RuntimeError(f"Frontier trie contract mismatch: {actual}")
    metadata = {
        "strategy": "frontier_all_system_prompt_response_sids",
        "source": str(source),
        "source_sha256": sha256_file(source),
        "provenance": str(provenance),
        "provenance_sha256": sha256_file(provenance),
        "records": records,
        "task_counts": dict(sorted(task_counts.items())),
        "unique_sids": len(sids),
        "domain_unique_sids": {
            domain: sum(1 for sid in sids if sid.domain == domain)
            for domain in ("video", "prod", "ad", "living")
        },
    }
    return trie.save(output_dir, metadata=metadata)


def _probe_rank(item: FrontierIndexedRecord) -> str:
    payload = f"{item.line_number}:{item.source_sha256}".encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def build_frontier_probe(
    source: Path,
    provenance: Path,
    output_path: Path,
    *,
    count_each: int = FRONTIER_PROBE_ROWS_PER_TASK,
    enforce_contract: bool = True,
) -> dict[str, Any]:
    """Select deterministic recommendation and item-text-to-SID probe rows."""

    source = Path(source).resolve()
    provenance = Path(provenance).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing Frontier probe: {output_path}")
    candidates: dict[str, list[FrontierIndexedRecord]] = {
        "recommend": [],
        "text_to_sid": [],
    }
    for item in iter_frontier_records(source, provenance):
        if item.task not in {"recommendation", "item_text_to_sid"}:
            continue
        suffix = final_answer_suffix(item.record.response)
        sids = find_sids(suffix)
        if len(sids) != 1:
            raise ValueError(
                f"Frontier probe source line {item.line_number} ({item.task}) "
                f"must contain exactly one final SID, got {len(sids)}."
            )
        key = "recommend" if item.task == "recommendation" else "text_to_sid"
        candidates[key].append(item)
    if enforce_contract and len(candidates["recommend"]) != FRONTIER_RECOMMEND_ROWS:
        raise RuntimeError("Frontier recommendation candidate count changed.")
    selected: list[tuple[str, FrontierIndexedRecord]] = []
    for key in ("recommend", "text_to_sid"):
        pool = sorted(candidates[key], key=_probe_rank)
        if len(pool) < count_each:
            raise RuntimeError(f"Not enough Frontier {key} rows for probe: {len(pool)}")
        selected.extend((key, item) for item in pool[:count_each])

    rows: list[dict[str, Any]] = []
    for key, item in selected:
        suffix = final_answer_suffix(item.record.response)
        sid_matches = find_sids(suffix)
        sid = sid_matches[0]
        rows.append(
            {
                "task": key,
                "system": item.record.system,
                "instruction": normalize_prompt_mode(item.record.prompt),
                "target": direct_response(item.record.response),
                "target_sid": sid.render(),
                "source_file": str(source),
                "source_line": item.line_number,
                "source_row_sha256": item.source_sha256,
            }
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    digest = hashlib.sha256()
    try:
        with temporary.open("wb") as handle:
            for row in rows:
                raw = (
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                ).encode("utf-8")
                handle.write(raw)
                digest.update(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if temporary.exists():
            temporary.unlink()
    manifest = {
        "schema_version": 1,
        "kind": "frontier_fixed_probe",
        "source": _file_record(source),
        "provenance": _file_record(provenance),
        "path": str(output_path),
        "size": output_path.stat().st_size,
        "sha256": digest.hexdigest(),
        "records": len(rows),
        "task_counts": dict(Counter(row["task"] for row in rows)),
        "selection": {
            "algorithm": "sha256(f'{candidate_line}:{raw_record_sha256}') ascending",
            "count_each": count_each,
            "prompt_mode": "/think|/no_think -> /no_think",
            "target": "direct empty-think wrapper plus final SID",
        },
    }
    _json_dump(output_path.parent / "probe_manifest.json", manifest)
    return manifest


def write_source_lock_report(source: Path, provenance: Path, output_path: Path) -> dict[str, Any]:
    report = build_source_lock_report(source, provenance)
    _json_dump(Path(output_path), report)
    return report
