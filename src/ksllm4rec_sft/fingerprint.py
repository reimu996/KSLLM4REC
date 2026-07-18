"""Deterministic fingerprint of files that can change training behavior."""

from __future__ import annotations

import hashlib
from pathlib import Path

from .data import sha256_file
from .profiles import BASELINE_PROFILE, SFTProfile, get_profile


EXPLICIT_PATHS = BASELINE_PROFILE.fingerprint_paths


def implementation_fingerprint(
    project_root: Path, profile: str | SFTProfile = BASELINE_PROFILE
) -> dict:
    project_root = project_root.resolve()
    selected = get_profile(profile)
    explicit_paths = (
        EXPLICIT_PATHS if selected is BASELINE_PROFILE else selected.fingerprint_paths
    )
    paths = [project_root / relative for relative in explicit_paths]
    paths.extend(sorted((project_root / "scripts/sft").glob("*.sh")))
    paths.extend(sorted((project_root / "scripts/sft").glob("*.py")))
    for relative in selected.fingerprint_script_dirs:
        paths.extend(
            sorted(
                path for path in (project_root / relative).rglob("*") if path.is_file()
            )
        )
    paths.extend(sorted((project_root / "src/ksllm4rec_sft").glob("*.py")))
    unique_paths = sorted(set(paths))
    digest = hashlib.sha256(b"ksllm4rec-sft-implementation-v1\0")
    if selected is not BASELINE_PROFILE:
        profile_bytes = selected.name.encode("utf-8")
        digest.update(len(profile_bytes).to_bytes(8, "big"))
        digest.update(profile_bytes)
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
