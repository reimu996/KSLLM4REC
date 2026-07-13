"""Deterministic fingerprint of files that can change training behavior."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .data import sha256_file


EXPLICIT_PATHS = (
    "configs/sft/artifacts.lock.json",
    "configs/sft/constraints.txt",
    "configs/sft/environment.lock.txt",
    "configs/sft/onereason_lora_focal_item.yaml",
)


def implementation_fingerprint(project_root: Path) -> dict:
    project_root = project_root.resolve()
    paths = [project_root / relative for relative in EXPLICIT_PATHS]
    paths.extend(sorted((project_root / "scripts/sft").glob("*.sh")))
    paths.extend(sorted((project_root / "scripts/sft").glob("*.py")))
    paths.extend(sorted((project_root / "src/ksllm4rec_sft").glob("*.py")))
    unique_paths = sorted(set(paths))
    digest = hashlib.sha256(b"ksllm4rec-sft-implementation-v1\0")
    files = {}
    for path in unique_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Fingerprint input is missing: {path}")
        relative = path.relative_to(project_root).as_posix()
        file_sha = sha256_file(path)
        files[relative] = file_sha
        relative_bytes = relative.encode("utf-8")
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(bytes.fromhex(file_sha))
    return {"sha256": digest.hexdigest(), "files": files}
