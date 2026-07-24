from __future__ import annotations

import unittest

from ksllm4rec_rloo_dapo.kv_cache import (
    GIB,
    KVCacheGeometry,
    estimate_kv_cache_bytes,
    fork_dynamic_cache,
    resolve_active_sequences,
)


class FakeCache:
    def __init__(self) -> None:
        self.rows = ["a", "b", "c"]

    def batch_select_indices(self, indices) -> None:
        self.rows = [self.rows[int(index)] for index in indices.tolist()]


class KVCacheTest(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = KVCacheGeometry(28, 8, 128, 2)

    def test_qwen_geometry_is_112_kib_per_sequence_token(self) -> None:
        self.assertEqual(self.geometry.bytes_per_sequence_token, 114_688)

    def test_observed_worst_case_fits_twelve_gib(self) -> None:
        value = estimate_kv_cache_bytes(
            self.geometry, base_rows=8, active_rows=16, sequence_length=2_929 + 32
        )
        self.assertLessEqual(value, 12 * GIB)
        self.assertEqual(
            resolve_active_sequences(
                self.geometry,
                base_rows=8,
                requested_active_rows=16,
                sequence_length=2_929 + 32,
                budget_gib=12.0,
            ),
            16,
        )

    def test_active_rows_reduce_under_a_small_budget(self) -> None:
        selected = resolve_active_sequences(
            self.geometry,
            base_rows=1,
            requested_active_rows=16,
            sequence_length=2_000,
            budget_gib=1.0,
        )
        self.assertIn(selected, (1, 2, 4))

    def test_cache_fork_does_not_mutate_the_base(self) -> None:
        base = FakeCache()
        forked = fork_dynamic_cache(base, [0, 0, 2])
        self.assertEqual(base.rows, ["a", "b", "c"])
        self.assertEqual(forked.rows, ["a", "a", "c"])


if __name__ == "__main__":
    unittest.main()
