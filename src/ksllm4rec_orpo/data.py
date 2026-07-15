"""Competition-row parsing and lossless direct-answer normalization."""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Iterable, Iterator

from ksllm4rec_sft.data import SourceRecord, iter_source_records


COMMON_SENSE_SYSTEM = "你是一个非常聪明的助手，请直接遵循指示作答。"
EMPTY_THINK = "<think>\n</think>"
MODE_SUFFIX_RE = re.compile(r"(?:/no_think|/think)\s*$")
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
SID_RE = re.compile(
    r"<\|(?P<domain>video|prod|ad|living)_begin\|>"
    r"<s_a_(?P<a>\d+)><s_b_(?P<b>\d+)><s_c_(?P<c>\d+)>"
)


class PairTask(StrEnum):
    RECOMMEND = "recommend"
    TEXT_TO_SID = "text_to_sid"
    SID_TO_TEXT = "sid_to_text"
    USER_INTEREST = "user_interest"
    CEVAL = "ceval"


EXPECTED_TASK_COUNTS = {
    PairTask.RECOMMEND: 18_651,
    PairTask.TEXT_TO_SID: 5_197,
    PairTask.SID_TO_TEXT: 4_487,
    PairTask.USER_INTEREST: 2_792,
    PairTask.CEVAL: 1_578,
}
EXPECTED_RECORDS = sum(EXPECTED_TASK_COUNTS.values())


@dataclass(frozen=True, order=True)
class Sid:
    domain: str
    a: int
    b: int
    c: int

    @classmethod
    def parse(cls, value: str) -> "Sid":
        match = SID_RE.fullmatch(value)
        if match is None:
            raise ValueError(f"Not a complete competition SID: {value!r}")
        return cls(
            domain=match.group("domain"),
            a=int(match.group("a")),
            b=int(match.group("b")),
            c=int(match.group("c")),
        )

    def render(self) -> str:
        return (
            f"<|{self.domain}_begin|><s_a_{self.a}>"
            f"<s_b_{self.b}><s_c_{self.c}>"
        )

    @property
    def ab_key(self) -> tuple[str, int, int]:
        return self.domain, self.a, self.b

    @property
    def a_key(self) -> tuple[str, int]:
        return self.domain, self.a


@dataclass(frozen=True)
class IndexedRecord:
    line_number: int
    source_sha256: str
    record: SourceRecord
    task: PairTask


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def source_row_sha256(record: SourceRecord) -> str:
    payload = json.dumps(
        [record.system, record.prompt, record.response],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return sha256_text(payload)


def final_answer_suffix(response: str) -> str:
    """Return bytes after the first closing think tag without trimming them."""

    before, marker, suffix = response.partition("</think>")
    if not marker or "<think>" not in before:
        raise ValueError("Response must contain one <think>...</think> block.")
    if "<think>" in suffix or "</think>" in suffix:
        raise ValueError("Response contains more than one think block.")
    if not suffix.strip():
        raise ValueError("Response has no final answer after </think>.")
    return suffix


def direct_response(response: str) -> str:
    """Remove think content while preserving the exact post-think suffix."""

    return EMPTY_THINK + final_answer_suffix(response)


def normalize_prompt_mode(prompt: str) -> str:
    base = MODE_SUFFIX_RE.sub("", prompt).rstrip()
    return f"{base}/no_think"


def find_sids(value: str) -> tuple[Sid, ...]:
    return tuple(Sid.parse(match.group(0)) for match in SID_RE.finditer(value))


def final_answer_text(response: str) -> str:
    return final_answer_suffix(response).strip()


def _recommend_systems(records: Iterable[SourceRecord]) -> set[str]:
    counts = Counter(record.system for record in records)
    systems = {system for system, count in counts.items() if count >= 3_000}
    if len(systems) != 6 or sum(counts[system] for system in systems) != 18_651:
        raise RuntimeError(
            "Could not recover the six fixed recommendation templates: "
            f"systems={len(systems)}, rows={sum(counts[system] for system in systems)}"
        )
    return systems


def classify_record(record: SourceRecord, recommend_systems: set[str]) -> PairTask:
    if record.system in recommend_systems:
        return PairTask.RECOMMEND
    if record.system == "":
        return PairTask.USER_INTEREST
    if record.system == COMMON_SENSE_SYSTEM:
        return PairTask.CEVAL

    answer_sids = find_sids(final_answer_suffix(record.response))
    prompt_sids = find_sids(record.prompt)
    if len(answer_sids) == 1:
        return PairTask.TEXT_TO_SID
    if prompt_sids and not answer_sids:
        return PairTask.SID_TO_TEXT
    raise ValueError(
        "Unclassified competition row: "
        f"system={record.system[:80]!r}, prompt_sids={len(prompt_sids)}, "
        f"answer_sids={len(answer_sids)}"
    )


def load_indexed_records(path: Path) -> list[IndexedRecord]:
    records = list(iter_source_records(path))
    if len(records) != EXPECTED_RECORDS:
        raise RuntimeError(
            f"Expected {EXPECTED_RECORDS} source rows, found {len(records)}."
        )
    recommend_systems = _recommend_systems(records)
    indexed = [
        IndexedRecord(
            line_number=index,
            source_sha256=source_row_sha256(record),
            record=record,
            task=classify_record(record, recommend_systems),
        )
        for index, record in enumerate(records, start=1)
    ]
    counts = Counter(item.task for item in indexed)
    if counts != Counter(EXPECTED_TASK_COUNTS):
        raise RuntimeError(
            f"Task counts do not match the frozen dataset contract: {dict(counts)}"
        )
    return indexed


def iter_pair_rows(path: Path) -> Iterator[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid pair JSON at line {line_number}: {exc}") from exc
            required = {"instruction", "input", "chosen", "rejected", "system"}
            if not isinstance(value, dict) or set(value) != required:
                raise ValueError(f"Invalid pair schema at line {line_number}.")
            if value["input"] != "" or value["chosen"] == value["rejected"]:
                raise ValueError(f"Invalid pair content at line {line_number}.")
            if not all(isinstance(value[key], str) for key in required):
                raise ValueError(f"Pair fields must be strings at line {line_number}.")
            yield value
