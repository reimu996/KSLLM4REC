"""Compact baseline SID prefix tree backed by CSR-like NumPy arrays."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .orpo_data import Sid


DOMAIN_ORDER = ("video", "prod", "ad", "living")
DOMAIN_TO_INDEX = {domain: index for index, domain in enumerate(DOMAIN_ORDER)}
MAX_COMPONENT = 8_191

_COMPONENT_BITS = 13
_B_SHIFT = _COMPONENT_BITS
_A_SHIFT = _COMPONENT_BITS * 2
_DOMAIN_SHIFT = _COMPONENT_BITS * 3
_COMPONENT_MASK = (1 << _COMPONENT_BITS) - 1

_ARRAY_DTYPES = {
    "domain_a_offsets": np.dtype(np.int64),
    "a_values": np.dtype(np.uint16),
    "da_b_offsets": np.dtype(np.int64),
    "b_values": np.dtype(np.uint16),
    "dab_c_offsets": np.dtype(np.int64),
    "c_values": np.dtype(np.uint16),
}
_ARRAY_FILENAMES = {name: f"{name}.npy" for name in _ARRAY_DTYPES}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_domain(domain: str) -> int:
    try:
        return DOMAIN_TO_INDEX[domain]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"SID domain must be one of {DOMAIN_ORDER}, got {domain!r}."
        ) from exc


def _validate_component(name: str, value: int) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"SID component {name} must be an integer, got {value!r}.")
    result = int(value)
    if result < 0 or result > MAX_COMPONENT:
        raise ValueError(
            f"SID component {name} must be within [0, {MAX_COMPONENT}], got {result}."
        )
    return result


def _pack_sid(sid: Sid) -> np.uint64:
    domain_index = _validate_domain(sid.domain)
    a = _validate_component("a", sid.a)
    b = _validate_component("b", sid.b)
    c = _validate_component("c", sid.c)
    return np.uint64(
        (domain_index << _DOMAIN_SHIFT) | (a << _A_SHIFT) | (b << _B_SHIFT) | c
    )


def _canonical_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    if metadata is None:
        return {}
    if not isinstance(metadata, Mapping):
        raise TypeError("Trie metadata must be a mapping.")
    encoded = json.dumps(
        dict(metadata),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    decoded = json.loads(encoded)
    if not isinstance(
        decoded, dict
    ):  # pragma: no cover - Mapping always encodes as object
        raise TypeError("Trie metadata must encode as a JSON object.")
    return decoded


def _validate_offsets(
    offsets: np.ndarray,
    *,
    name: str,
    expected_size: int,
    child_count: int,
    require_nonempty_parents: bool,
) -> None:
    if offsets.ndim != 1 or offsets.dtype != np.dtype(np.int64):
        raise ValueError(f"{name} must be a one-dimensional int64 array.")
    if offsets.size != expected_size:
        raise ValueError(
            f"{name} must have shape [{expected_size}], got {list(offsets.shape)}."
        )
    if int(offsets[0]) != 0 or int(offsets[-1]) != child_count:
        raise ValueError(
            f"{name} must start at 0 and end at child count {child_count}."
        )
    differences = offsets[1:] - offsets[:-1]
    if bool(np.any(differences < 0)):
        raise ValueError(f"{name} must be nondecreasing.")
    if require_nonempty_parents and bool(np.any(differences == 0)):
        raise ValueError(f"{name} must assign at least one child to every parent.")


def _validate_values(values: np.ndarray, *, name: str) -> None:
    if values.ndim != 1 or values.dtype != np.dtype(np.uint16):
        raise ValueError(f"{name} must be a one-dimensional uint16 array.")
    if values.size and int(values.max()) > MAX_COMPONENT:
        raise ValueError(f"{name} contains a value above {MAX_COMPONENT}.")


def _validate_strict_sibling_order(
    values: np.ndarray, offsets: np.ndarray, *, name: str
) -> None:
    if values.size < 2:
        return
    invalid = values[1:] <= values[:-1]
    boundary_starts = offsets[1:-1]
    if boundary_starts.size:
        invalid[boundary_starts - 1] = False
    if bool(np.any(invalid)):
        raise ValueError(f"{name} siblings must be strictly increasing.")


class SidPrefixTrie:
    """Immutable SID trie with domain/a/b/c represented as CSR child arrays."""

    def __init__(
        self,
        *,
        domain_a_offsets: np.ndarray,
        a_values: np.ndarray,
        da_b_offsets: np.ndarray,
        b_values: np.ndarray,
        dab_c_offsets: np.ndarray,
        c_values: np.ndarray,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        self.domain_a_offsets = np.asanyarray(domain_a_offsets)
        self.a_values = np.asanyarray(a_values)
        self.da_b_offsets = np.asanyarray(da_b_offsets)
        self.b_values = np.asanyarray(b_values)
        self.dab_c_offsets = np.asanyarray(dab_c_offsets)
        self.c_values = np.asanyarray(c_values)
        self.metadata = _canonical_metadata(metadata)
        self.validate()
        for array in self.arrays.values():
            array.setflags(write=False)

    @classmethod
    def from_sids(cls, sids: Iterable[Sid]) -> "SidPrefixTrie":
        """Build a deduplicated trie from an arbitrary-order SID iterable."""

        packed = np.fromiter((_pack_sid(sid) for sid in sids), dtype=np.uint64)
        if packed.size == 0:
            raise ValueError("Cannot build a SID prefix tree from an empty iterable.")
        keys = np.unique(packed)

        domains = (keys >> np.uint64(_DOMAIN_SHIFT)).astype(np.uint8)
        a = ((keys >> np.uint64(_A_SHIFT)) & _COMPONENT_MASK).astype(np.uint16)
        b = ((keys >> np.uint64(_B_SHIFT)) & _COMPONENT_MASK).astype(np.uint16)
        c = (keys & _COMPONENT_MASK).astype(np.uint16)

        a_starts_mask = np.empty(keys.size, dtype=np.bool_)
        a_starts_mask[0] = True
        a_starts_mask[1:] = (domains[1:] != domains[:-1]) | (a[1:] != a[:-1])
        a_starts = np.flatnonzero(a_starts_mask)

        b_starts_mask = a_starts_mask.copy()
        b_starts_mask[1:] |= b[1:] != b[:-1]
        b_starts = np.flatnonzero(b_starts_mask)

        a_values = a[a_starts]
        b_values = b[b_starts]

        domain_a_counts = np.bincount(
            domains[a_starts], minlength=len(DOMAIN_ORDER)
        ).astype(np.int64)
        domain_a_offsets = np.empty(len(DOMAIN_ORDER) + 1, dtype=np.int64)
        domain_a_offsets[0] = 0
        np.cumsum(domain_a_counts, out=domain_a_offsets[1:])

        a_group_for_b = np.searchsorted(a_starts, b_starts, side="right") - 1
        b_counts = np.bincount(a_group_for_b, minlength=a_starts.size).astype(np.int64)
        da_b_offsets = np.empty(a_starts.size + 1, dtype=np.int64)
        da_b_offsets[0] = 0
        np.cumsum(b_counts, out=da_b_offsets[1:])

        dab_c_offsets = np.empty(b_starts.size + 1, dtype=np.int64)
        dab_c_offsets[:-1] = b_starts
        dab_c_offsets[-1] = keys.size

        return cls(
            domain_a_offsets=domain_a_offsets,
            a_values=a_values,
            da_b_offsets=da_b_offsets,
            b_values=b_values,
            dab_c_offsets=dab_c_offsets,
            c_values=c,
        )

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        return {name: getattr(self, name) for name in _ARRAY_DTYPES}

    @property
    def leaf_count(self) -> int:
        return int(self.c_values.size)

    @property
    def counts(self) -> dict[str, Any]:
        domain_counts: dict[str, dict[str, int]] = {}
        for domain_index, domain in enumerate(DOMAIN_ORDER):
            a_start = int(self.domain_a_offsets[domain_index])
            a_stop = int(self.domain_a_offsets[domain_index + 1])
            b_start = int(self.da_b_offsets[a_start])
            b_stop = int(self.da_b_offsets[a_stop])
            c_start = int(self.dab_c_offsets[b_start])
            c_stop = int(self.dab_c_offsets[b_stop])
            domain_counts[domain] = {
                "a_nodes": a_stop - a_start,
                "ab_nodes": b_stop - b_start,
                "leaves": c_stop - c_start,
            }
        return {
            "a_nodes": int(self.a_values.size),
            "ab_nodes": int(self.b_values.size),
            "leaves": self.leaf_count,
            "by_domain": domain_counts,
        }

    def validate(self, *, expected_leaf_count: int | None = None) -> None:
        """Validate dtypes, shapes, CSR structure, ranges, and sibling ordering."""

        _validate_values(self.a_values, name="a_values")
        _validate_values(self.b_values, name="b_values")
        _validate_values(self.c_values, name="c_values")
        if self.c_values.size == 0:
            raise ValueError("SID prefix tree must contain at least one leaf.")
        if expected_leaf_count is not None and self.leaf_count != expected_leaf_count:
            raise ValueError(
                "SID prefix tree leaf count mismatch: "
                f"expected {expected_leaf_count}, got {self.leaf_count}."
            )

        _validate_offsets(
            self.domain_a_offsets,
            name="domain_a_offsets",
            expected_size=len(DOMAIN_ORDER) + 1,
            child_count=int(self.a_values.size),
            require_nonempty_parents=False,
        )
        _validate_offsets(
            self.da_b_offsets,
            name="da_b_offsets",
            expected_size=int(self.a_values.size) + 1,
            child_count=int(self.b_values.size),
            require_nonempty_parents=True,
        )
        _validate_offsets(
            self.dab_c_offsets,
            name="dab_c_offsets",
            expected_size=int(self.b_values.size) + 1,
            child_count=int(self.c_values.size),
            require_nonempty_parents=True,
        )

        for domain_index, domain in enumerate(DOMAIN_ORDER):
            start = int(self.domain_a_offsets[domain_index])
            stop = int(self.domain_a_offsets[domain_index + 1])
            values = self.a_values[start:stop]
            if values.size > 1 and bool(np.any(values[1:] <= values[:-1])):
                raise ValueError(
                    f"a_values for domain {domain!r} must be strictly increasing."
                )
        _validate_strict_sibling_order(
            self.b_values, self.da_b_offsets, name="b_values"
        )
        _validate_strict_sibling_order(
            self.c_values, self.dab_c_offsets, name="c_values"
        )

    @staticmethod
    def _find_child(
        values: np.ndarray, offsets: np.ndarray, parent: int, target: int
    ) -> int | None:
        start = int(offsets[parent])
        stop = int(offsets[parent + 1])
        relative = int(np.searchsorted(values[start:stop], np.uint16(target)))
        index = start + relative
        if index >= stop or int(values[index]) != target:
            return None
        return index

    def allowed_a(self, domain: str) -> np.ndarray:
        domain_index = _validate_domain(domain)
        start = int(self.domain_a_offsets[domain_index])
        stop = int(self.domain_a_offsets[domain_index + 1])
        return self.a_values[start:stop]

    def allowed_b(self, domain: str, a: int) -> np.ndarray:
        domain_index = _validate_domain(domain)
        a_value = _validate_component("a", a)
        a_index = self._find_child(
            self.a_values, self.domain_a_offsets, domain_index, a_value
        )
        if a_index is None:
            return self.b_values[:0]
        start = int(self.da_b_offsets[a_index])
        stop = int(self.da_b_offsets[a_index + 1])
        return self.b_values[start:stop]

    def allowed_c(self, domain: str, a: int, b: int) -> np.ndarray:
        domain_index = _validate_domain(domain)
        a_value = _validate_component("a", a)
        b_value = _validate_component("b", b)
        a_index = self._find_child(
            self.a_values, self.domain_a_offsets, domain_index, a_value
        )
        if a_index is None:
            return self.c_values[:0]
        b_index = self._find_child(self.b_values, self.da_b_offsets, a_index, b_value)
        if b_index is None:
            return self.c_values[:0]
        start = int(self.dab_c_offsets[b_index])
        stop = int(self.dab_c_offsets[b_index + 1])
        return self.c_values[start:stop]

    def contains(self, sid: Sid) -> bool:
        a = _validate_component("a", sid.a)
        b = _validate_component("b", sid.b)
        c = _validate_component("c", sid.c)
        allowed = self.allowed_c(sid.domain, a, b)
        relative = int(np.searchsorted(allowed, np.uint16(c)))
        return relative < allowed.size and int(allowed[relative]) == c

    def save(
        self,
        output_dir: Path,
        *,
        metadata: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Save arrays and an integrity manifest to a new directory."""

        self.validate()
        manifest_metadata = _canonical_metadata(
            self.metadata if metadata is None else metadata
        )
        output_dir = Path(output_dir)
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        if output_dir.exists():
            raise FileExistsError(output_dir)
        temporary = Path(
            tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent)
        )
        try:
            array_manifest: dict[str, dict[str, Any]] = {}
            for name, array in self.arrays.items():
                path = temporary / _ARRAY_FILENAMES[name]
                np.save(path, array, allow_pickle=False)
                array_manifest[name] = {
                    "file": path.name,
                    "dtype": str(array.dtype),
                    "shape": list(array.shape),
                    "size": path.stat().st_size,
                    "sha256": _sha256(path),
                }

            manifest = {
                "schema_version": 1,
                "domain_order": list(DOMAIN_ORDER),
                "counts": self.counts,
                "arrays": array_manifest,
                "metadata": manifest_metadata,
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True)
                + "\n",
                encoding="utf-8",
            )
            self.load(
                temporary,
                mmap_mode=None,
                expected_leaf_count=self.leaf_count,
            )
            os.rename(temporary, output_dir)
            return manifest
        except BaseException:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    @classmethod
    def load(
        cls,
        output_dir: Path,
        *,
        mmap_mode: str | None = "r",
        expected_leaf_count: int | None = None,
    ) -> "SidPrefixTrie":
        """Load a saved trie after verifying its manifest and every array."""

        output_dir = Path(output_dir)
        manifest_path = output_dir / "manifest.json"
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Cannot read trie manifest: {manifest_path}.") from exc

        required = {"schema_version", "domain_order", "counts", "arrays", "metadata"}
        if not isinstance(manifest, dict) or set(manifest) != required:
            raise ValueError("Trie manifest has an invalid top-level schema.")
        if manifest["schema_version"] != 1:
            raise ValueError("Unsupported trie manifest schema version.")
        if manifest["domain_order"] != list(DOMAIN_ORDER):
            raise ValueError("Trie manifest has a different domain order.")
        if not isinstance(manifest["arrays"], dict) or set(manifest["arrays"]) != set(
            _ARRAY_DTYPES
        ):
            raise ValueError("Trie manifest has an invalid array inventory.")

        arrays: dict[str, np.ndarray] = {}
        for name, expected_dtype in _ARRAY_DTYPES.items():
            entry = manifest["arrays"][name]
            expected_entry_keys = {"file", "dtype", "shape", "size", "sha256"}
            if not isinstance(entry, dict) or set(entry) != expected_entry_keys:
                raise ValueError(f"Trie manifest entry for {name} is invalid.")
            if entry["file"] != _ARRAY_FILENAMES[name]:
                raise ValueError(f"Trie manifest file name for {name} is invalid.")
            path = output_dir / entry["file"]
            if not path.is_file():
                raise ValueError(f"Trie array is missing: {path}.")
            if path.stat().st_size != entry["size"] or _sha256(path) != entry["sha256"]:
                raise ValueError(f"Trie array integrity check failed: {name}.")
            try:
                array = np.load(path, mmap_mode=mmap_mode, allow_pickle=False)
            except (OSError, ValueError) as exc:
                raise ValueError(f"Cannot load trie array: {name}.") from exc
            if array.dtype != expected_dtype or entry["dtype"] != str(expected_dtype):
                raise ValueError(f"Trie array dtype mismatch: {name}.")
            if list(array.shape) != entry["shape"]:
                raise ValueError(f"Trie array shape mismatch: {name}.")
            arrays[name] = array

        trie = cls(**arrays, metadata=manifest["metadata"])
        if manifest["counts"] != trie.counts:
            raise ValueError("Trie manifest counts do not match the loaded arrays.")
        trie.validate(expected_leaf_count=expected_leaf_count)
        return trie
