from __future__ import annotations

import unittest
from dataclasses import dataclass

from ksllm4rec_rloo_dapo.sampling import (
    DeterministicGroupStream,
    SourceCursor,
    append_effective_groups,
)


@dataclass(frozen=True)
class Group:
    group_id: str


class SamplingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.groups = [Group(f"g{index:03d}") for index in range(32)]

    def test_stream_resume_is_exact(self) -> None:
        stream = DeterministicGroupStream(self.groups, seed=42)
        seen: set[str] = set()
        first = stream.take_unique(8, seen)
        cursor = stream.cursor
        second = stream.take_unique(8, seen)

        resumed = DeterministicGroupStream(self.groups, seed=42, cursor=cursor)
        resumed_seen = {group.group_id for group in first}
        self.assertEqual(second, resumed.take_unique(8, resumed_seen))

    def test_window_cannot_reuse_a_prompt(self) -> None:
        stream = DeterministicGroupStream(self.groups, seed=42)
        seen: set[str] = set()
        selected = stream.take_unique(32, seen)
        self.assertEqual(len({group.group_id for group in selected}), 32)
        with self.assertRaisesRegex(RuntimeError, "reuse"):
            stream.take_unique(1, seen)

    def test_overflow_is_deterministically_truncated(self) -> None:
        result = append_effective_groups(range(29), range(29, 33), target=32)
        self.assertEqual(result.retained, tuple(range(32)))
        self.assertEqual(result.overflow, (32,))


if __name__ == "__main__":
    unittest.main()
