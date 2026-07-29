"""Build or verify the frozen Frontier text-to-SID group artifact for V1.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import yaml


_HERE = Path(__file__).resolve()
_PROJECT_ROOT = _HERE.parents[3]
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify(output: Path) -> dict[str, object]:
    from ksllm4rec_dapo_anchor_multitask_v1_1 import contract
    from ksllm4rec_dapo_anchor_multitask_v1_1.data import (
        load_text_to_sid_target_groups,
    )

    groups_path = output / "groups.jsonl"
    manifest_path = output / "data_manifest.json"
    groups = load_text_to_sid_target_groups(groups_path)
    actual = {
        "groups_sha256": _sha256(groups_path),
        "manifest_sha256": _sha256(manifest_path),
    }
    expected = {
        "groups_sha256": contract.TEXT_TO_SID_GROUPS_SHA256,
        "manifest_sha256": contract.TEXT_TO_SID_MANIFEST_SHA256,
    }
    if actual != expected:
        raise RuntimeError(
            f"Text-to-SID artifact fingerprint mismatch: {actual!r}"
        )
    return {
        "groups": len(groups),
        **actual,
        "output": str(output),
    }


def main() -> int:
    parser = argparse.ArgumentParser(prog="prepare-dapo-anchor-multitask-v1.1-e2-data")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    from ksllm4rec_dapo_anchor_multitask_v1_1.config import validate_config
    from ksllm4rec_dapo_anchor_multitask_v1_1.data import (
        build_frontier_text_to_sid_groups,
        write_text_to_sid_artifact,
    )

    validate_config(config)
    output = args.output_dir.resolve()
    expected = Path(config["output"]["text_to_sid_groups_dir"]).resolve()
    if output != expected:
        raise ValueError(f"Text group output must be {expected}.")

    if output.exists():
        report = _verify(output)
        report["event"] = "text_to_sid_groups_already_exist_verified"
    else:
        build = build_frontier_text_to_sid_groups(
            Path(config["data"]["source"]),
            Path(config["data"]["provenance"]),
        )
        write_text_to_sid_artifact(
            build,
            output,
            source=Path(config["data"]["source"]),
            provenance=Path(config["data"]["provenance"]),
        )
        report = _verify(output)
        report["event"] = "text_to_sid_groups_created"
    print(json.dumps(report, ensure_ascii=False, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
