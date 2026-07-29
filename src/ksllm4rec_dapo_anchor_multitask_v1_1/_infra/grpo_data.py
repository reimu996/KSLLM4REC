"""Lossless recommendation prompt grouping without precomputed completions."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

from .orpo_data import (
    IndexedRecord,
    PairTask,
    Sid,
    find_sids,
    final_answer_suffix,
    load_indexed_records,
    normalize_prompt_mode,
)
from .sft_data import SourceRecord, iter_source_records

from .grpo_contract import (
    EXPECTED_BASELINE_UNIQUE_SIDS,
    EXPECTED_DOMAIN_SIDS,
    EXPECTED_RECOMMEND_GROUPS,
    EXPECTED_RECOMMEND_ROWS,
    EXPECTED_UNIQUE_RECOMMEND_POSITIVES,
    MAX_SID_COMPONENT,
    POSITIVE_SET_SIZE_DISTRIBUTION,
)


@dataclass(frozen=True)
class RecommendationGroup:
    group_id: str
    system: str
    prompt: str
    positive_sids: tuple[str, ...]
    source_lines: tuple[int, ...]

    def to_dict(self) -> dict:
        return {
            "schema_version": 1,
            "group_id": self.group_id,
            "system": self.system,
            "prompt": self.prompt,
            "positive_sids": list(self.positive_sids),
            "source_lines": list(self.source_lines),
        }


def _group_id(system: str, prompt: str) -> str:
    payload = json.dumps([system, prompt], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def group_recommendation_records(
    records: Iterable[IndexedRecord], *, enforce_contract: bool = True
) -> list[RecommendationGroup]:
    grouped: dict[str, dict] = {}
    recommend_rows = 0
    for item in records:
        if item.task != PairTask.RECOMMEND:
            continue
        recommend_rows += 1
        prompt = normalize_prompt_mode(item.record.prompt)
        sids = find_sids(final_answer_suffix(item.record.response))
        if len(sids) != 1:
            raise ValueError(
                f"Recommendation source line {item.line_number} must contain one final SID."
            )
        state = grouped.setdefault(
            prompt,
            {"systems": set(), "positive_sids": set(), "source_lines": []},
        )
        state["systems"].add(item.record.system)
        state["positive_sids"].add(sids[0].render())
        state["source_lines"].append(item.line_number)

    groups = []
    for prompt, state in grouped.items():
        if len(state["systems"]) != 1:
            raise ValueError(
                f"Recommendation prompt maps to multiple systems: {prompt[:120]!r}"
            )
        system = next(iter(state["systems"]))
        positives = tuple(sorted(state["positive_sids"]))
        groups.append(
            RecommendationGroup(
                group_id=_group_id(system, prompt),
                system=system,
                prompt=prompt,
                positive_sids=positives,
                source_lines=tuple(state["source_lines"]),
            )
        )
    groups.sort(key=lambda item: item.group_id)

    if enforce_contract:
        distribution = Counter(len(item.positive_sids) for item in groups)
        if recommend_rows != EXPECTED_RECOMMEND_ROWS:
            raise RuntimeError(
                f"Expected {EXPECTED_RECOMMEND_ROWS} recommendation rows, got {recommend_rows}."
            )
        if len(groups) != EXPECTED_RECOMMEND_GROUPS:
            raise RuntimeError(
                f"Expected {EXPECTED_RECOMMEND_GROUPS} recommendation groups, got {len(groups)}."
            )
        if sum(len(item.positive_sids) for item in groups) != EXPECTED_RECOMMEND_ROWS:
            raise RuntimeError(
                "Recommendation positives were lost or duplicated during grouping."
            )
        unique_positives = {sid for item in groups for sid in item.positive_sids}
        if len(unique_positives) != EXPECTED_UNIQUE_RECOMMEND_POSITIVES:
            raise RuntimeError(
                "Recommendation unique-positive count differs from the frozen contract."
            )
        if dict(sorted(distribution.items())) != POSITIVE_SET_SIZE_DISTRIBUTION:
            raise RuntimeError(
                "Recommendation positive-set distribution differs from the frozen contract."
            )
    return groups


def build_recommendation_groups(source: Path) -> list[RecommendationGroup]:
    return group_recommendation_records(load_indexed_records(source))


def write_groups(groups: Iterable[RecommendationGroup], path: Path) -> dict:
    rows = list(groups)
    path.parent.mkdir(parents=True, exist_ok=False)
    digest = hashlib.sha256()
    with path.open("wb") as handle:
        for item in rows:
            raw = (
                json.dumps(item.to_dict(), ensure_ascii=False, separators=(",", ":"))
                + "\n"
            ).encode("utf-8")
            handle.write(raw)
            digest.update(raw)
    return {
        "path": str(path),
        "rows": len(rows),
        "size": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def iter_groups(path: Path) -> Iterator[RecommendationGroup]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            value = json.loads(line)
            required = {
                "schema_version",
                "group_id",
                "system",
                "prompt",
                "positive_sids",
                "source_lines",
            }
            if not isinstance(value, dict) or set(value) != required:
                raise ValueError(f"Invalid group schema at line {line_number}.")
            if value["schema_version"] != 1:
                raise ValueError(f"Unsupported group schema at line {line_number}.")
            item = RecommendationGroup(
                group_id=value["group_id"],
                system=value["system"],
                prompt=value["prompt"],
                positive_sids=tuple(value["positive_sids"]),
                source_lines=tuple(value["source_lines"]),
            )
            if item.group_id != _group_id(item.system, item.prompt):
                raise ValueError(f"Group ID mismatch at line {line_number}.")
            yield item


def extract_sids_from_records(records: Iterable[SourceRecord]) -> set[Sid]:
    """Extract every complete SID from the three baseline string fields."""

    sids: set[Sid] = set()
    for record in records:
        for field in (record.system, record.prompt, record.response):
            for sid in find_sids(field):
                if max(sid.a, sid.b, sid.c) > MAX_SID_COMPONENT:
                    raise ValueError(
                        f"SID component exceeds {MAX_SID_COMPONENT}: {sid.render()}"
                    )
                sids.add(sid)
    if not sids:
        raise RuntimeError("No complete SID was found in the baseline source.")
    return sids


def scan_baseline_sids(
    source: Path, *, enforce_contract: bool = True
) -> tuple[set[Sid], dict[str, int]]:
    """Scan baseline JSONL once and return its unique legal-SID universe."""

    sids = extract_sids_from_records(iter_source_records(source))
    by_domain = Counter(sid.domain for sid in sids)
    counts = {domain: int(by_domain[domain]) for domain in sorted(by_domain)}
    if enforce_contract:
        if len(sids) != EXPECTED_BASELINE_UNIQUE_SIDS:
            raise RuntimeError(
                "Baseline SID count differs from the frozen contract: "
                f"expected={EXPECTED_BASELINE_UNIQUE_SIDS}, actual={len(sids)}"
            )
        if counts != dict(sorted(EXPECTED_DOMAIN_SIDS.items())):
            raise RuntimeError(
                "Baseline SID domain counts differ from the frozen contract: "
                f"expected={EXPECTED_DOMAIN_SIDS}, actual={counts}"
            )
    return sids, counts
