"""Deterministic candidate-pair construction for all competition tasks."""

from __future__ import annotations

import json
import os
import re
import tempfile
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from ksllm4rec_sft.data import SourceRecord, sha256_file
from ksllm4rec_sft.manifest import atomic_write_json

from .data import (
    EMPTY_THINK,
    EXPECTED_RECORDS,
    EXPECTED_TASK_COUNTS,
    IndexedRecord,
    PairTask,
    Sid,
    direct_response,
    final_answer_suffix,
    find_sids,
    load_indexed_records,
    normalize_prompt_mode,
    sha256_text,
)


RAW_DOMAIN_TO_TOKEN = {
    "goods": "prod",
    "video/video": "video",
    "video/ad": "ad",
    "live": "living",
}
OPTION_RE = re.compile(r"\(([ABCD])\)")
CANDIDATE_SCHEMA_VERSION = 1


def _numeric_path_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.stem), path.name
    except ValueError:
        return 2**31 - 1, path.name


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":"))
                    + "\n"
                )
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp_name, path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return count


def _append_unique(
    mapping: dict[Any, list[Any]], key: Any, value: Any, limit: int
) -> None:
    values = mapping[key]
    if len(values) >= limit:
        return
    if value not in values:
        values.append(value)


@dataclass(frozen=True)
class SidCatalogReport:
    files: int
    rows_scanned: int
    invalid_rows: int
    target_ab_keys: int
    target_a_keys: int
    stored_ab_candidates: int
    stored_a_candidates: int
    stored_domain_candidates: int


@dataclass
class SidCatalog:
    by_ab: dict[tuple[str, int, int], list[Sid]]
    by_a: dict[tuple[str, int], list[Sid]]
    by_domain: dict[str, list[Sid]]
    report: SidCatalogReport

    def candidates(
        self,
        chosen: Sid,
        positive_set: set[str],
        *,
        limit: int = 8,
    ) -> tuple[int, str, list[Sid]]:
        def eligible(values: Iterable[Sid]) -> list[Sid]:
            unique = {
                value
                for value in values
                if value != chosen and value.render() not in positive_set
            }
            return sorted(unique)[:limit]

        tier1 = eligible(self.by_ab.get(chosen.ab_key, ()))
        if tier1:
            return 1, "same_ab_wrong_c", tier1
        tier2 = eligible(self.by_a.get(chosen.a_key, ()))
        if tier2:
            return 2, "same_a_wrong_bc", tier2
        tier3 = eligible(self.by_domain.get(chosen.domain, ()))
        if tier3:
            return 3, "same_domain_wrong_sid", tier3
        raise RuntimeError(f"No negative SID candidate for {chosen.render()}")


def build_sid_catalog(
    pid2sid_dir: Path,
    targets: Iterable[Sid],
    *,
    per_key_limit: int = 64,
    domain_limit: int = 256,
) -> SidCatalog:
    import numpy as np
    import pyarrow.compute as pc
    import pyarrow.dataset as ds

    target_list = list(targets)
    target_ab = {sid.ab_key for sid in target_list}
    target_a = {sid.a_key for sid in target_list}
    token_domains = tuple(RAW_DOMAIN_TO_TOKEN.values())
    domain_to_id = {domain: index for index, domain in enumerate(token_domains)}

    def encode_a(domain: str, a: int) -> int:
        return (domain_to_id[domain] << 40) | (a << 20)

    def encode_ab(domain: str, a: int, b: int) -> int:
        return encode_a(domain, a) | b

    target_a_codes = {encode_a(*key): key for key in target_a}
    target_ab_codes = {encode_ab(*key): key for key in target_ab}
    by_ab: dict[tuple[str, int, int], list[Sid]] = defaultdict(list)
    by_a: dict[tuple[str, int], list[Sid]] = defaultdict(list)
    by_domain: dict[str, list[Sid]] = defaultdict(list)
    files = sorted(pid2sid_dir.glob("*.parquet"))
    if len(files) != 198:
        raise RuntimeError(f"Expected 198 Pid2Sid parquet files, found {len(files)}")

    rows_scanned = 0
    invalid_rows = 0
    scanner = ds.dataset([str(path) for path in files], format="parquet").scanner(
        columns=["domain", "sid_three"], batch_size=131_072
    )
    for batch in scanner.to_batches():
        domain_array = batch.column(0)
        sid_array = batch.column(1)
        batch_rows = len(batch)
        rows_scanned += batch_rows
        lengths = pc.list_value_length(sid_array).to_numpy(
            zero_copy_only=False
        )
        if sid_array.null_count or not bool(np.all(lengths == 3)):
            raise RuntimeError(
                "Pid2Sid contains null or non-three-element SID rows; the frozen "
                "dataset requires an explicit slow-path audit."
            )
        floats = pc.list_flatten(sid_array).to_numpy(
            zero_copy_only=False
        ).reshape(batch_rows, 3)
        finite = np.isfinite(floats).all(axis=1)
        safe_floats = np.where(finite[:, None], floats, 0.0)
        integers = safe_floats.astype(np.int64)
        integral = np.equal(safe_floats, integers).all(axis=1)
        in_range = ((integers >= 0) & (integers < (1 << 20))).all(axis=1)
        raw_domains = domain_array.to_numpy(zero_copy_only=False)
        domain_ids = np.full(batch_rows, -1, dtype=np.int64)
        for raw_domain, token_domain in RAW_DOMAIN_TO_TOKEN.items():
            domain_ids[raw_domains == raw_domain] = domain_to_id[token_domain]
        valid = finite & integral & in_range & (domain_ids >= 0)
        invalid_rows += int((~valid).sum())

        a_codes = (domain_ids << 40) | (integers[:, 0] << 20)
        ab_codes = a_codes | integers[:, 1]
        active_ab = np.fromiter(
            (
                code
                for code, key in target_ab_codes.items()
                if len(by_ab[key]) < per_key_limit
            ),
            dtype=np.int64,
        )
        if active_ab.size:
            indices = np.flatnonzero(valid & np.isin(ab_codes, active_ab))
            for index in indices:
                key = target_ab_codes[int(ab_codes[index])]
                sid = Sid(key[0], *map(int, integers[index]))
                _append_unique(by_ab, key, sid, per_key_limit)

        active_a = np.fromiter(
            (
                code
                for code, key in target_a_codes.items()
                if len(by_a[key]) < per_key_limit
            ),
            dtype=np.int64,
        )
        if active_a.size:
            indices = np.flatnonzero(valid & np.isin(a_codes, active_a))
            for index in indices:
                key = target_a_codes[int(a_codes[index])]
                sid = Sid(key[0], *map(int, integers[index]))
                _append_unique(by_a, key, sid, per_key_limit)

        for token_domain, domain_id in domain_to_id.items():
            if len(by_domain[token_domain]) >= domain_limit:
                continue
            indices = np.flatnonzero(valid & (domain_ids == domain_id))
            for index in indices:
                sid = Sid(token_domain, *map(int, integers[index]))
                _append_unique(by_domain, token_domain, sid, domain_limit)
                if len(by_domain[token_domain]) >= domain_limit:
                    break

    report = SidCatalogReport(
        files=len(files),
        rows_scanned=rows_scanned,
        invalid_rows=invalid_rows,
        target_ab_keys=len(target_ab),
        target_a_keys=len(target_a),
        stored_ab_candidates=sum(map(len, by_ab.values())),
        stored_a_candidates=sum(map(len, by_a.values())),
        stored_domain_candidates=sum(map(len, by_domain.values())),
    )
    if rows_scanned != 35_914_095:
        raise RuntimeError(
            f"Expected 35,914,095 Pid2Sid rows, scanned {rows_scanned}."
        )
    return SidCatalog(dict(by_ab), dict(by_a), dict(by_domain), report)


@dataclass(frozen=True)
class CaptionEntry:
    sid: Sid
    response: str
    source: str


@dataclass(frozen=True)
class CaptionCatalogReport:
    baseline_entries: int
    explorer_files_scanned: int
    explorer_rows_scanned: int
    explorer_target_a_found: int
    requested_target_a: int


@dataclass
class CaptionCatalog:
    baseline_by_ab: dict[tuple[str, int, int], list[CaptionEntry]]
    baseline_by_a: dict[tuple[str, int], list[CaptionEntry]]
    baseline_by_domain: dict[str, list[CaptionEntry]]
    explorer_by_a: dict[tuple[str, int], list[CaptionEntry]]
    explorer_by_domain: dict[str, list[CaptionEntry]]
    report: CaptionCatalogReport

    def candidates(
        self, chosen_sid: Sid, chosen_response: str, *, limit: int = 8
    ) -> tuple[int, str, list[CaptionEntry]]:
        def eligible(values: Iterable[CaptionEntry]) -> list[CaptionEntry]:
            result: list[CaptionEntry] = []
            seen: set[str] = set()
            for value in values:
                if value.sid == chosen_sid or value.response == chosen_response:
                    continue
                digest = sha256_text(value.response)
                if digest in seen:
                    continue
                seen.add(digest)
                result.append(value)
                if len(result) == limit:
                    break
            return result

        tier1 = eligible(self.baseline_by_ab.get(chosen_sid.ab_key, ()))
        if tier1:
            return 1, "same_ab_wrong_caption", tier1
        tier2 = eligible(self.baseline_by_a.get(chosen_sid.a_key, ()))
        if tier2:
            return 2, "same_a_wrong_caption", tier2
        explorer_tier2 = eligible(self.explorer_by_a.get(chosen_sid.a_key, ()))
        if explorer_tier2:
            return 2, "explorer_same_a_wrong_caption", explorer_tier2
        explorer_tier3 = eligible(
            self.explorer_by_domain.get(chosen_sid.domain, ())
        )
        if explorer_tier3:
            return 3, "explorer_same_domain_wrong_caption", explorer_tier3
        tier3 = eligible(self.baseline_by_domain.get(chosen_sid.domain, ()))
        if tier3:
            return 3, "same_domain_wrong_caption", tier3
        raise RuntimeError(f"No caption negative for {chosen_sid.render()}")


def _iter_explorer_rows(directory: Path) -> Iterator[tuple[Path, int, SourceRecord]]:
    import orjson

    files = sorted(directory.glob("*.jsonl"), key=_numeric_path_key)
    if len(files) != 31:
        raise RuntimeError(
            f"Expected 31 Explorer SID-to-text files, found {len(files)}"
        )
    for path in files:
        with path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                value = orjson.loads(line)
                if (
                    not isinstance(value, list)
                    or len(value) != 1
                    or not isinstance(value[0], dict)
                ):
                    raise ValueError(f"Invalid Explorer row at {path}:{line_number}")
                row = value[0]
                yield path, line_number, SourceRecord(
                    row["system"], row["prompt"], row["response"]
                )


def build_caption_catalog(
    indexed: list[IndexedRecord], explorer_sid_to_text_dir: Path
) -> CaptionCatalog:
    baseline_by_ab: dict[tuple[str, int, int], list[CaptionEntry]] = defaultdict(list)
    baseline_by_a: dict[tuple[str, int], list[CaptionEntry]] = defaultdict(list)
    baseline_by_domain: dict[str, list[CaptionEntry]] = defaultdict(list)
    sid_to_text_records = [
        item for item in indexed if item.task == PairTask.SID_TO_TEXT
    ]
    for item in sid_to_text_records:
        prompt_sids = find_sids(item.record.prompt)
        if len(prompt_sids) != 1:
            raise ValueError(
                f"SID-to-text row {item.line_number} must contain exactly one input SID."
            )
        sid = prompt_sids[0]
        entry = CaptionEntry(sid, direct_response(item.record.response), "baseline")
        _append_unique(baseline_by_ab, sid.ab_key, entry, 64)
        _append_unique(baseline_by_a, sid.a_key, entry, 64)
        _append_unique(baseline_by_domain, sid.domain, entry, 256)

    needed_a: set[tuple[str, int]] = set()
    for item in sid_to_text_records:
        sid = find_sids(item.record.prompt)[0]
        chosen_response = direct_response(item.record.response)
        alternatives = [
            entry
            for entry in baseline_by_a.get(sid.a_key, ())
            if entry.sid != sid and entry.response != chosen_response
        ]
        if not alternatives:
            needed_a.add(sid.a_key)

    explorer_by_a: dict[tuple[str, int], list[CaptionEntry]] = defaultdict(list)
    explorer_by_domain: dict[str, list[CaptionEntry]] = defaultdict(list)
    scanned_files: set[Path] = set()
    scanned_rows = 0
    for path, _line_number, record in _iter_explorer_rows(
        explorer_sid_to_text_dir
    ):
        scanned_files.add(path)
        scanned_rows += 1
        prompt_sids = find_sids(record.prompt)
        if len(prompt_sids) != 1:
            continue
        sid = prompt_sids[0]
        response = direct_response(record.response)
        entry = CaptionEntry(sid, response, f"explorer:{path.name}")
        if sid.a_key in needed_a:
            _append_unique(explorer_by_a, sid.a_key, entry, 8)
        _append_unique(explorer_by_domain, sid.domain, entry, 64)
        if (
            all(len(explorer_by_a.get(key, ())) >= 2 for key in needed_a)
            and all(len(explorer_by_domain.get(domain, ())) >= 64 for domain in RAW_DOMAIN_TO_TOKEN.values())
        ):
            break

    return CaptionCatalog(
        baseline_by_ab=dict(baseline_by_ab),
        baseline_by_a=dict(baseline_by_a),
        baseline_by_domain=dict(baseline_by_domain),
        explorer_by_a=dict(explorer_by_a),
        explorer_by_domain=dict(explorer_by_domain),
        report=CaptionCatalogReport(
            baseline_entries=len(sid_to_text_records),
            explorer_files_scanned=len(scanned_files),
            explorer_rows_scanned=scanned_rows,
            explorer_target_a_found=sum(bool(explorer_by_a.get(key)) for key in needed_a),
            requested_target_a=len(needed_a),
        ),
    )


def _recommend_positive_sets(
    indexed: Iterable[IndexedRecord],
) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for item in indexed:
        if item.task != PairTask.RECOMMEND:
            continue
        key = normalize_prompt_mode(item.record.prompt)
        sids = find_sids(final_answer_suffix(item.record.response))
        if len(sids) != 1:
            raise ValueError(
                f"Recommendation row {item.line_number} must have one final SID."
            )
        groups[key].add(sids[0].render())
    if len(groups) != 6_378 or sum(map(len, groups.values())) != 18_651:
        raise RuntimeError(
            "Normalized recommendation positive groups do not match the frozen contract: "
            f"groups={len(groups)}, pairs={sum(map(len, groups.values()))}"
        )
    return groups


def _text_positive_sets(
    indexed: Iterable[IndexedRecord],
) -> dict[str, set[str]]:
    groups: dict[str, set[str]] = defaultdict(set)
    for item in indexed:
        if item.task != PairTask.TEXT_TO_SID:
            continue
        key = normalize_prompt_mode(item.record.prompt)
        sids = find_sids(final_answer_suffix(item.record.response))
        if len(sids) != 1:
            raise ValueError(
                f"Text-to-SID row {item.line_number} must have one final SID."
            )
        groups[key].add(sids[0].render())
    return groups


def _replace_single_sid(response: str, old: Sid, new: Sid) -> str:
    occurrences = find_sids(final_answer_suffix(response))
    if occurrences != (old,):
        raise ValueError("Expected exactly the chosen SID in the final answer.")
    return response.replace(old.render(), new.render(), 1)


def _ceval_candidates(chosen: str) -> list[str]:
    suffix = final_answer_suffix(chosen)
    matches = list(OPTION_RE.finditer(suffix))
    if len(matches) != 1:
        raise ValueError("CEval response must contain exactly one parenthesized option.")
    match = matches[0]
    correct = match.group(1)
    prefix = chosen[: -len(suffix)]
    return [
        prefix + suffix[: match.start(1)] + option + suffix[match.end(1) :]
        for option in "ABCD"
        if option != correct
    ]


def _user_rejected_map(
    indexed: list[IndexedRecord],
) -> dict[int, tuple[str, int]]:
    users = [item for item in indexed if item.task == PairTask.USER_INTEREST]
    metadata: list[tuple[set[str], set[str], str, type[Any]]] = []
    for item in users:
        prompt_sids = {sid.render() for sid in find_sids(item.record.prompt)}
        response = direct_response(item.record.response)
        response_sids = {sid.render() for sid in find_sids(response)}
        parsed = json.loads(final_answer_suffix(response).strip())
        metadata.append((prompt_sids, response_sids, response, type(parsed)))

    result: dict[int, tuple[str, int]] = {}
    for index, item in enumerate(users):
        prompt_sids, _chosen_sids, chosen_response, chosen_type = metadata[index]
        for jump in range(1, len(users)):
            candidate_index = (index + jump) % len(users)
            _, candidate_sids, candidate_response, candidate_type = metadata[
                candidate_index
            ]
            if (
                candidate_type is chosen_type
                and candidate_response != chosen_response
                and candidate_sids
                and candidate_sids.isdisjoint(prompt_sids)
            ):
                result[item.line_number] = (
                    candidate_response,
                    users[candidate_index].line_number,
                )
                break
        if item.line_number not in result:
            raise RuntimeError(
                f"No disjoint user-interest rejected response for line {item.line_number}."
            )
    return result


def _candidate_row(
    item: IndexedRecord,
    *,
    chosen: str,
    candidates: list[str],
    tier: int,
    reason: str,
    positive_set: set[str],
    requires_model_mining: bool,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if not candidates or any(candidate == chosen for candidate in candidates):
        raise RuntimeError(f"Invalid rejected candidates at source line {item.line_number}.")
    if not chosen.startswith(EMPTY_THINK) or any(
        not candidate.startswith(EMPTY_THINK) for candidate in candidates
    ):
        raise RuntimeError("All chosen/rejected responses must use direct no-think.")
    candidate_hash = sha256_text(
        json.dumps(candidates, ensure_ascii=False, separators=(",", ":"))
    )
    row = {
        "schema_version": CANDIDATE_SCHEMA_VERSION,
        "source_line": item.line_number,
        "source_row_sha256": item.source_sha256,
        "task": item.task.value,
        "system": item.record.system,
        "instruction": normalize_prompt_mode(item.record.prompt),
        "chosen": chosen,
        "rejected_candidates": candidates,
        "negative_tier": tier,
        "negative_reason": reason,
        "positive_set": sorted(positive_set),
        "positive_set_sha256": sha256_text(
            json.dumps(sorted(positive_set), ensure_ascii=False)
        ),
        "candidate_set_sha256": candidate_hash,
        "requires_model_mining": requires_model_mining,
        "original_final_suffix_sha256": sha256_text(
            final_answer_suffix(item.record.response)
        ),
    }
    if extra:
        row["extra"] = extra
    return row


def build_candidate_rows(
    indexed: list[IndexedRecord],
    sid_catalog: SidCatalog,
    caption_catalog: CaptionCatalog,
) -> Iterator[dict[str, Any]]:
    recommend_sets = _recommend_positive_sets(indexed)
    text_sets = _text_positive_sets(indexed)
    user_rejected = _user_rejected_map(indexed)

    for item in indexed:
        chosen = direct_response(item.record.response)
        if item.task == PairTask.RECOMMEND:
            chosen_sid = find_sids(final_answer_suffix(chosen))[0]
            positive_set = recommend_sets[normalize_prompt_mode(item.record.prompt)]
            tier, reason, sid_candidates = sid_catalog.candidates(
                chosen_sid, positive_set
            )
            candidates = [
                _replace_single_sid(chosen, chosen_sid, sid)
                for sid in sid_candidates
            ]
            yield _candidate_row(
                item,
                chosen=chosen,
                candidates=candidates,
                tier=tier,
                reason=reason,
                positive_set=positive_set,
                requires_model_mining=len(candidates) > 1,
            )
        elif item.task == PairTask.TEXT_TO_SID:
            chosen_sid = find_sids(final_answer_suffix(chosen))[0]
            positive_set = text_sets[normalize_prompt_mode(item.record.prompt)]
            tier, reason, sid_candidates = sid_catalog.candidates(
                chosen_sid, positive_set
            )
            candidates = [
                _replace_single_sid(chosen, chosen_sid, sid)
                for sid in sid_candidates
            ]
            yield _candidate_row(
                item,
                chosen=chosen,
                candidates=candidates,
                tier=tier,
                reason=reason,
                positive_set=positive_set,
                requires_model_mining=len(candidates) > 1,
            )
        elif item.task == PairTask.SID_TO_TEXT:
            chosen_sid = find_sids(item.record.prompt)[0]
            tier, reason, entries = caption_catalog.candidates(chosen_sid, chosen)
            yield _candidate_row(
                item,
                chosen=chosen,
                candidates=[entry.response for entry in entries],
                tier=tier,
                reason=reason,
                positive_set={chosen},
                requires_model_mining=False,
                extra={
                    "candidate_sources": [entry.source for entry in entries],
                    "candidate_sids": [entry.sid.render() for entry in entries],
                },
            )
        elif item.task == PairTask.USER_INTEREST:
            rejected, rejected_source_line = user_rejected[item.line_number]
            yield _candidate_row(
                item,
                chosen=chosen,
                candidates=[rejected],
                tier=1,
                reason="cross_user_disjoint_sids",
                positive_set={chosen},
                requires_model_mining=False,
                extra={"rejected_source_line": rejected_source_line},
            )
        elif item.task == PairTask.CEVAL:
            candidates = _ceval_candidates(chosen)
            yield _candidate_row(
                item,
                chosen=chosen,
                candidates=candidates,
                tier=1,
                reason="wrong_option",
                positive_set={chosen},
                requires_model_mining=True,
            )
        else:
            raise AssertionError(f"Unhandled task: {item.task}")


def prepare_pair_candidates(
    source_path: Path,
    pid2sid_dir: Path,
    explorer_sid_to_text_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source_path = source_path.resolve()
    output_dir = output_dir.resolve()
    indexed = load_indexed_records(source_path)
    sid_targets = [
        find_sids(final_answer_suffix(item.record.response))[0]
        for item in indexed
        if item.task in (PairTask.RECOMMEND, PairTask.TEXT_TO_SID)
    ]
    sid_catalog = build_sid_catalog(pid2sid_dir.resolve(), sid_targets)
    caption_catalog = build_caption_catalog(
        indexed, explorer_sid_to_text_dir.resolve()
    )
    candidates_path = output_dir / "candidates.jsonl"
    rows = list(build_candidate_rows(indexed, sid_catalog, caption_catalog))
    if len(rows) != EXPECTED_RECORDS:
        raise RuntimeError(f"Expected {EXPECTED_RECORDS} candidate rows, got {len(rows)}")
    count = _atomic_jsonl(candidates_path, rows)
    task_counts = Counter(row["task"] for row in rows)
    expected = {task.value: count for task, count in EXPECTED_TASK_COUNTS.items()}
    if dict(task_counts) != expected:
        raise RuntimeError(f"Candidate task counts mismatch: {dict(task_counts)}")
    tier_counts = Counter(
        (row["task"], row["negative_tier"], row["negative_reason"])
        for row in rows
    )
    report = {
        "schema_version": 1,
        "status": "passed",
        "source_path": str(source_path),
        "source_size": source_path.stat().st_size,
        "source_sha256": sha256_file(source_path),
        "records": count,
        "task_counts": dict(sorted(task_counts.items())),
        "tier_counts": {
            f"{task}|tier{tier}|{reason}": value
            for (task, tier, reason), value in sorted(tier_counts.items())
        },
        "model_mining_rows": sum(row["requires_model_mining"] for row in rows),
        "candidates_path": str(candidates_path),
        "candidates_size": candidates_path.stat().st_size,
        "candidates_sha256": sha256_file(candidates_path),
        "pid2sid_dir": str(pid2sid_dir.resolve()),
        "sid_catalog": asdict(sid_catalog.report),
        "explorer_sid_to_text_dir": str(explorer_sid_to_text_dir.resolve()),
        "caption_catalog": asdict(caption_catalog.report),
    }
    atomic_write_json(output_dir / "candidate_report.json", report)
    return report
