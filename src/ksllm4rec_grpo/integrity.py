"""Input hashing and immutable artifact records."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_file(path: Path, expected_sha256: str) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(
            f"Input SHA256 mismatch for {path}: expected={expected_sha256}, actual={actual}"
        )
    return {"path": str(path.resolve()), "size": path.stat().st_size, "sha256": actual}
