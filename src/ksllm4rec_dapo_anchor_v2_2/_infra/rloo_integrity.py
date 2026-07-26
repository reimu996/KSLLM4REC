"""Content-addressed records used by the independent RLOO workflow."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable


_CHUNK_BYTES = 1024 * 1024


def sha256_file(path: Path) -> str:
    """Return the SHA256 of one regular file."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    raise TypeError(f"Value is not JSON serializable: {type(value).__name__}")


def canonical_json_bytes(value: Any) -> bytes:
    """Encode a JSON-compatible value with one deterministic representation."""

    def reject_non_finite(node: Any) -> None:
        if isinstance(node, float) and not math.isfinite(node):
            raise ValueError("Canonical JSON cannot contain NaN or infinity.")
        if isinstance(node, dict):
            for key, child in node.items():
                if not isinstance(key, str):
                    raise TypeError("Canonical JSON object keys must be strings.")
                reject_non_finite(child)
        elif isinstance(node, (list, tuple)):
            for child in node:
                reject_non_finite(child)

    reject_non_finite(value)
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=_json_default,
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def file_record(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    """Describe one file and optionally enforce a caller-supplied digest."""

    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if expected_sha256 is not None and actual != expected_sha256:
        raise RuntimeError(
            f"SHA256 mismatch for {path}: expected={expected_sha256}, actual={actual}"
        )
    return {
        "path": str(path.resolve()),
        "size": path.stat().st_size,
        "sha256": actual,
    }


def require_file(path: Path, expected_sha256: str) -> dict[str, Any]:
    """Compatibility spelling for a file whose digest is already frozen."""

    return file_record(path, expected_sha256)


def snapshot_directory(
    root: Path,
    *,
    relative_paths: Iterable[str | Path] | None = None,
    exclude: Iterable[str | Path] = (),
    require_nonempty: bool = True,
) -> dict[str, dict[str, Any]]:
    """Hash an exact regular-file set below ``root``.

    The returned keys are POSIX relative paths. Symlinks are rejected so a
    checkpoint cannot silently depend on data outside its atomic directory.
    """

    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(root)
    excluded = {Path(value).as_posix() for value in exclude}
    if relative_paths is None:
        candidates = sorted(path for path in root.rglob("*") if path.is_file())
    else:
        candidates = [root / Path(value) for value in relative_paths]
    records: dict[str, dict[str, Any]] = {}
    for path in candidates:
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        if path.is_symlink():
            raise RuntimeError(f"Symlinks are forbidden in immutable trees: {path}")
        if not path.is_file():
            raise FileNotFoundError(path)
        if relative in records:
            raise ValueError(f"Duplicate immutable-tree path: {relative}")
        records[relative] = {
            "size": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    if require_nonempty and not records:
        raise RuntimeError(f"Immutable tree contains no files: {root}")
    return dict(sorted(records.items()))


def verify_directory_snapshot(
    root: Path,
    expected: dict[str, dict[str, Any]],
    *,
    exclude: Iterable[str | Path] = (),
) -> dict[str, dict[str, Any]]:
    """Require the directory's file names, sizes, and SHA256 values to match."""

    actual = snapshot_directory(root, exclude=exclude, require_nonempty=False)
    expected_names = set(expected)
    actual_names = set(actual)
    if expected_names != actual_names:
        missing = sorted(expected_names - actual_names)
        extra = sorted(actual_names - expected_names)
        raise RuntimeError(
            f"Immutable tree file set mismatch: missing={missing}, extra={extra}"
        )
    for name in sorted(expected):
        record = expected[name]
        if set(record) != {"size", "sha256"}:
            raise ValueError(f"Invalid immutable-tree record for {name}.")
        if actual[name] != record:
            raise RuntimeError(
                f"Immutable tree SHA256/size mismatch for {name}: "
                f"expected={record}, actual={actual[name]}"
            )
    return actual
