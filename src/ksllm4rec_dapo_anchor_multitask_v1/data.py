"""Typed source groups for recommendation and item-text-to-SID training."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Iterable, Iterator

from ._infra.frontier_data import (
    FrontierIndexedRecord,
    iter_frontier_records,
)
from ._infra.grpo_data import RecommendationGroup, iter_groups
from ._infra.orpo_data import (
    Sid,
    final_answer_suffix,
    find_sids,
    normalize_prompt_mode,
)


FRONTIER_TEXT_TO_SID_ROWS = 13_353
FRONTIER_TEXT_TO_SID_GROUPS = 10_597
FRONTIER_TEXT_TO_SID_DUPLICATE_ROWS = 2_756
FRONTIER_TEXT_TO_SID_UNIQUE_SIDS = 10_577


class SidTask(StrEnum):
    """A prompt task whose target is one or more complete SIDs."""

    RECOMMENDATION = "recommendation"
    ITEM_TEXT_TO_SID = "item_text_to_sid"


def canonical_group_id(system: str, prompt: str) -> str:
    """Match the stable group identity used by the Frontier data builder."""

    payload = json.dumps([system, prompt], ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SidTargetGroup:
    """One normalized prompt and its complete set of accepted SID targets."""

    task: SidTask
    group_id: str
    system: str
    prompt: str
    positive_sids: tuple[str, ...]
    source_lines: tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.task, SidTask):
            raise TypeError("task must be a SidTask.")
        if self.group_id != canonical_group_id(self.system, self.prompt):
            raise ValueError("group_id does not match system and prompt.")
        if not self.positive_sids:
            raise ValueError("A SID target group needs at least one positive SID.")
        if tuple(sorted(set(self.positive_sids))) != self.positive_sids:
            raise ValueError("positive_sids must be unique and sorted.")
        for value in self.positive_sids:
            Sid.parse(value)
        if not self.source_lines:
            raise ValueError("A SID target group needs at least one source line.")
        if tuple(sorted(set(self.source_lines))) != self.source_lines:
            raise ValueError("source_lines must be unique and sorted.")

    @property
    def positives(self) -> tuple[Sid, ...]:
        return tuple(Sid.parse(value) for value in self.positive_sids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "task": self.task.value,
            "group_id": self.group_id,
            "system": self.system,
            "prompt": self.prompt,
            "positive_sids": list(self.positive_sids),
            "source_lines": list(self.source_lines),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "SidTargetGroup":
        required = {
            "schema_version",
            "task",
            "group_id",
            "system",
            "prompt",
            "positive_sids",
            "source_lines",
        }
        if not isinstance(value, dict) or set(value) != required:
            raise ValueError("Invalid SID target group schema.")
        if value["schema_version"] != 1:
            raise ValueError("Unsupported SID target group schema version.")
        return cls(
            task=SidTask(value["task"]),
            group_id=value["group_id"],
            system=value["system"],
            prompt=value["prompt"],
            positive_sids=tuple(value["positive_sids"]),
            source_lines=tuple(value["source_lines"]),
        )


@dataclass(frozen=True)
class TextToSidGroupBuild:
    """The normalized groups plus the counts needed to audit the contraction."""

    groups: tuple[SidTargetGroup, ...]
    raw_rows: int
    duplicate_rows: int
    unique_positive_sids: int
    variant_counts: tuple[tuple[str, int], ...]


def recommendation_target_group(group: RecommendationGroup) -> SidTargetGroup:
    """Adapt the existing immutable recommendation artifact without rewriting it."""

    return SidTargetGroup(
        task=SidTask.RECOMMENDATION,
        group_id=group.group_id,
        system=group.system,
        prompt=group.prompt,
        positive_sids=tuple(group.positive_sids),
        source_lines=tuple(group.source_lines),
    )


def load_recommendation_target_groups(path: Path) -> tuple[SidTargetGroup, ...]:
    """Read the frozen recommendation JSONL as the shared multitask group type."""

    groups = tuple(
        recommendation_target_group(group) for group in iter_groups(Path(path))
    )
    if len({group.group_id for group in groups}) != len(groups):
        raise ValueError("Recommendation group IDs must be unique.")
    return groups


def _text_target(item: FrontierIndexedRecord) -> tuple[str, str]:
    prompt = normalize_prompt_mode(item.record.prompt)
    sids = find_sids(final_answer_suffix(item.record.response))
    if len(sids) != 1:
        raise ValueError(
            f"Frontier item_text_to_sid line {item.line_number} must contain "
            f"exactly one final SID, got {len(sids)}."
        )
    return prompt, sids[0].render()


def group_text_to_sid_records(
    records: Iterable[FrontierIndexedRecord], *, enforce_contract: bool = True
) -> TextToSidGroupBuild:
    """Collapse direct/thinking variants by normalized ``(system, prompt)``.

    A normalized prompt is required to map to exactly one SID.  Conflicting
    supervision is rejected instead of being silently converted into a
    multi-target item-identification group.
    """

    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    variants: Counter[str] = Counter()
    raw_rows = 0
    for item in records:
        if item.task != SidTask.ITEM_TEXT_TO_SID.value:
            continue
        raw_rows += 1
        prompt, sid = _text_target(item)
        key = (item.record.system, prompt)
        state = grouped.setdefault(
            key,
            {"positive_sids": set(), "source_lines": [], "variants": Counter()},
        )
        state["positive_sids"].add(sid)
        state["source_lines"].append(item.line_number)
        variant = item.variant or "(none)"
        state["variants"][variant] += 1
        variants[variant] += 1

    conflicts = [
        (key, sorted(state["positive_sids"]))
        for key, state in grouped.items()
        if len(state["positive_sids"]) != 1
    ]
    if conflicts:
        (system, prompt), targets = conflicts[0]
        raise ValueError(
            "A normalized item_text_to_sid prompt maps to multiple SIDs: "
            f"system={system[:80]!r}, prompt={prompt[:120]!r}, targets={targets!r}."
        )

    groups = tuple(
        sorted(
            (
                SidTargetGroup(
                    task=SidTask.ITEM_TEXT_TO_SID,
                    group_id=canonical_group_id(system, prompt),
                    system=system,
                    prompt=prompt,
                    positive_sids=tuple(sorted(state["positive_sids"])),
                    source_lines=tuple(sorted(set(state["source_lines"]))),
                )
                for (system, prompt), state in grouped.items()
            ),
            key=lambda group: group.group_id,
        )
    )
    duplicate_rows = raw_rows - len(groups)
    unique_positive_sids = len(
        {sid for group in groups for sid in group.positive_sids}
    )
    result = TextToSidGroupBuild(
        groups=groups,
        raw_rows=raw_rows,
        duplicate_rows=duplicate_rows,
        unique_positive_sids=unique_positive_sids,
        variant_counts=tuple(sorted(variants.items())),
    )
    if enforce_contract:
        actual = {
            "raw_rows": result.raw_rows,
            "groups": len(result.groups),
            "duplicate_rows": result.duplicate_rows,
            "unique_positive_sids": result.unique_positive_sids,
            "groups_with_non_single_target": sum(
                len(group.positive_sids) != 1 for group in result.groups
            ),
        }
        expected = {
            "raw_rows": FRONTIER_TEXT_TO_SID_ROWS,
            "groups": FRONTIER_TEXT_TO_SID_GROUPS,
            "duplicate_rows": FRONTIER_TEXT_TO_SID_DUPLICATE_ROWS,
            "unique_positive_sids": FRONTIER_TEXT_TO_SID_UNIQUE_SIDS,
            "groups_with_non_single_target": 0,
        }
        if actual != expected:
            raise RuntimeError(f"Frontier item_text_to_sid contract mismatch: {actual}")
    return result


def build_frontier_text_to_sid_groups(
    source: Path, provenance: Path, *, enforce_contract: bool = True
) -> TextToSidGroupBuild:
    """Load byte-aligned Frontier rows and build normalized text target groups."""

    return group_text_to_sid_records(
        iter_frontier_records(Path(source), Path(provenance)),
        enforce_contract=enforce_contract,
    )


def iter_target_groups(path: Path) -> Iterator[SidTargetGroup]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            try:
                value = json.loads(line)
                group = SidTargetGroup.from_dict(value)
            except (json.JSONDecodeError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid SID target group at line {line_number}."
                ) from exc
            yield group


def load_text_to_sid_target_groups(path: Path) -> tuple[SidTargetGroup, ...]:
    groups = tuple(iter_target_groups(Path(path)))
    if len(groups) != FRONTIER_TEXT_TO_SID_GROUPS:
        raise RuntimeError(
            f"Expected {FRONTIER_TEXT_TO_SID_GROUPS} text groups, got {len(groups)}."
        )
    if any(group.task is not SidTask.ITEM_TEXT_TO_SID for group in groups):
        raise ValueError("Text-to-SID artifact contains another task.")
    if len({group.group_id for group in groups}) != len(groups):
        raise ValueError("Text-to-SID artifact contains duplicate group IDs.")
    return groups


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_text_to_sid_artifact(
    build: TextToSidGroupBuild,
    output_dir: Path,
    *,
    source: Path,
    provenance: Path,
) -> dict[str, Any]:
    """Atomically publish the deterministic normalized text-group artifact."""

    output = Path(output_dir).resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite text groups: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.tmp-", dir=output.parent)
    )
    try:
        groups_path = temporary / "groups.jsonl"
        digest = hashlib.sha256()
        with groups_path.open("wb") as handle:
            for group in build.groups:
                raw = (
                    json.dumps(
                        group.to_dict(),
                        ensure_ascii=False,
                        separators=(",", ":"),
                    )
                    + "\n"
                ).encode("utf-8")
                handle.write(raw)
                digest.update(raw)
            handle.flush()
            os.fsync(handle.fileno())
        manifest = {
            "schema_version": 1,
            "kind": "frontier_text_to_sid_groups",
            "source": {
                "path": str(Path(source).resolve()),
                "sha256": _sha256_file(Path(source)),
            },
            "provenance": {
                "path": str(Path(provenance).resolve()),
                "sha256": _sha256_file(Path(provenance)),
            },
            "raw_rows": build.raw_rows,
            "groups": len(build.groups),
            "duplicate_rows": build.duplicate_rows,
            "positive_edges": sum(len(group.positive_sids) for group in build.groups),
            "unique_positive_sids": build.unique_positive_sids,
            "multi_gt_groups": sum(
                len(group.positive_sids) > 1 for group in build.groups
            ),
            "variant_counts": dict(build.variant_counts),
            "groups_file": {
                "path": "groups.jsonl",
                "size": groups_path.stat().st_size,
                "sha256": digest.hexdigest(),
            },
        }
        manifest_path = temporary / "data_manifest.json"
        with manifest_path.open("w", encoding="utf-8") as handle:
            json.dump(
                manifest,
                handle,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


__all__ = [
    "SidTargetGroup",
    "SidTask",
    "TextToSidGroupBuild",
    "build_frontier_text_to_sid_groups",
    "canonical_group_id",
    "group_text_to_sid_records",
    "iter_target_groups",
    "load_recommendation_target_groups",
    "load_text_to_sid_target_groups",
    "write_text_to_sid_artifact",
]
