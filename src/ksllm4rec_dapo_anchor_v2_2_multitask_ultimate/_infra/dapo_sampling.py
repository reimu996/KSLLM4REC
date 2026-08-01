"""Deterministic source traversal and effective-group accumulation."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class SourceCursor:
    cycle_index: int
    offset: int

    def __post_init__(self) -> None:
        if self.cycle_index < 0 or self.offset < 0:
            raise ValueError("Source cursor fields must be non-negative.")


@dataclass(frozen=True)
class BufferAppendResult:
    retained: tuple[Any, ...]
    overflow: tuple[Any, ...]


class DeterministicGroupStream:
    def __init__(
        self,
        groups: Sequence[Any],
        *,
        seed: int,
        cursor: SourceCursor | None = None,
    ) -> None:
        self._groups = tuple(groups)
        if not self._groups:
            raise ValueError("groups must not be empty.")
        ids = [self._group_id(group) for group in self._groups]
        if len(set(ids)) != len(ids):
            raise ValueError("group_id values must be unique.")
        self._by_id = dict(zip(ids, self._groups, strict=True))
        self._seed = int(seed)
        value = cursor or SourceCursor(cycle_index=0, offset=0)
        if value.offset >= len(self._groups):
            raise ValueError("Source cursor offset is outside the dataset.")
        self._cycle = value.cycle_index
        self._offset = value.offset
        self._order = self._permutation(self._cycle)

    @staticmethod
    def _group_id(group: Any) -> str:
        value = getattr(group, "group_id", None)
        if not isinstance(value, str) or not value:
            raise ValueError("Every source group needs a non-empty group_id.")
        task = getattr(group, "task", None)
        if task is None:
            return value
        task_value = getattr(task, "value", task)
        if not isinstance(task_value, str) or not task_value:
            raise ValueError("Every multitask source group needs a non-empty task.")
        return f"{task_value}|{value}"

    def _permutation(self, cycle: int) -> tuple[str, ...]:
        prefix = f"source-order|{self._seed}|{cycle}|".encode("utf-8")
        return tuple(
            sorted(
                self._by_id,
                key=lambda group_id: hashlib.sha256(
                    prefix + group_id.encode("utf-8")
                ).digest(),
            )
        )

    @property
    def cursor(self) -> SourceCursor:
        return SourceCursor(cycle_index=self._cycle, offset=self._offset)

    def _next(self) -> Any:
        group_id = self._order[self._offset]
        self._offset += 1
        if self._offset == len(self._order):
            self._cycle += 1
            self._offset = 0
            self._order = self._permutation(self._cycle)
        return self._by_id[group_id]

    def take_unique(self, count: int, seen_ids: set[str]) -> tuple[Any, ...]:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer.")
        if len(seen_ids) + count > len(self._groups):
            raise RuntimeError("A window cannot reuse a source prompt.")
        selected: list[Any] = []
        reads = 0
        maximum_reads = len(self._groups) * 2
        while len(selected) < count and reads < maximum_reads:
            group = self._next()
            reads += 1
            group_id = self._group_id(group)
            if group_id in seen_ids:
                continue
            seen_ids.add(group_id)
            selected.append(group)
        if len(selected) != count:
            raise RuntimeError("Could not find enough unique prompts for the window.")
        return tuple(selected)


def append_effective_groups(
    existing: Sequence[Any],
    new_groups: Sequence[Any],
    *,
    target: int,
) -> BufferAppendResult:
    if isinstance(target, bool) or not isinstance(target, int) or target <= 0:
        raise ValueError("target must be a positive integer.")
    combined = tuple(existing) + tuple(new_groups)
    return BufferAppendResult(
        retained=combined[:target],
        overflow=combined[target:],
    )


__all__ = [
    "BufferAppendResult",
    "DeterministicGroupStream",
    "SourceCursor",
    "append_effective_groups",
]
