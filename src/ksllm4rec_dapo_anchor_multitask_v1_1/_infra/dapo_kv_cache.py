"""KV-cache sizing, active-row resolution, and safe cache forking."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Sequence


GIB = 1024**3


@dataclass(frozen=True)
class KVCacheGeometry:
    layers: int
    kv_heads: int
    head_dim: int
    dtype_bytes: int

    def __post_init__(self) -> None:
        if min(self.layers, self.kv_heads, self.head_dim, self.dtype_bytes) <= 0:
            raise ValueError("KV-cache geometry values must be positive.")

    @property
    def bytes_per_sequence_token(self) -> int:
        return 2 * self.layers * self.kv_heads * self.head_dim * self.dtype_bytes


def estimate_kv_cache_bytes(
    geometry: KVCacheGeometry,
    *,
    base_rows: int,
    active_rows: int,
    sequence_length: int,
) -> int:
    values = (base_rows, active_rows, sequence_length)
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 0 for value in values):
        raise ValueError("KV-cache row and length values must be non-negative integers.")
    return (
        (base_rows + active_rows)
        * sequence_length
        * geometry.bytes_per_sequence_token
    )


def resolve_active_sequences(
    geometry: KVCacheGeometry,
    *,
    base_rows: int,
    requested_active_rows: int,
    sequence_length: int,
    budget_gib: float,
    choices: Sequence[int] = (16, 8, 4, 2, 1),
) -> int:
    if requested_active_rows <= 0:
        raise ValueError("requested_active_rows must be positive.")
    budget_bytes = int(float(budget_gib) * GIB)
    if budget_bytes <= 0:
        raise ValueError("budget_gib must be positive.")
    allowed = [value for value in choices if 0 < int(value) <= requested_active_rows]
    for active_rows in allowed:
        if estimate_kv_cache_bytes(
            geometry,
            base_rows=base_rows,
            active_rows=int(active_rows),
            sequence_length=sequence_length,
        ) <= budget_bytes:
            return int(active_rows)
    return 0


def fork_dynamic_cache(cache: Any, indices: Sequence[int]) -> Any:
    values = [int(index) for index in indices]
    if not values or min(values) < 0:
        raise ValueError("cache fork indices must be non-empty and non-negative.")
    to_legacy = getattr(cache, "to_legacy_cache", None)
    from_legacy = getattr(type(cache), "from_legacy_cache", None)
    if callable(to_legacy) and callable(from_legacy):
        import torch

        legacy = to_legacy()
        if not legacy:
            raise ValueError("Cannot fork an empty cache.")
        device = legacy[0][0].device
        tensor = torch.tensor(values, dtype=torch.long, device=device)
        selected = tuple(
            (
                keys.index_select(0, tensor),
                values_tensor.index_select(0, tensor),
            )
            for keys, values_tensor in legacy
        )
        return from_legacy(selected)

    forked = copy.deepcopy(cache)
    selector = getattr(forked, "batch_select_indices", None)
    if not callable(selector):
        raise TypeError("Cache does not support batch_select_indices().")
    try:
        import torch

        device = None
        layers = getattr(forked, "layers", ())
        for layer in layers:
            keys = getattr(layer, "keys", None)
            if keys is not None and getattr(keys, "numel", lambda: 0)() > 0:
                device = keys.device
                break
        tensor = torch.tensor(values, dtype=torch.long, device=device)
        selector(tensor)
    except Exception:
        del forked
        raise
    return forked


__all__ = [
    "GIB",
    "KVCacheGeometry",
    "estimate_kv_cache_bytes",
    "fork_dynamic_cache",
    "resolve_active_sequences",
]
