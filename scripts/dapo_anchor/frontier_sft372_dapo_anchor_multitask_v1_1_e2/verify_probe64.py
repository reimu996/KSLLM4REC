"""Verify fixed-probe identity, policy bytes, trie membership, and Pass@64."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

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


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object.")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(prog="verify-dapo-anchor-multitask-v1.1-e2-probe64")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()

    from ksllm4rec_dapo_anchor_multitask_v1_1._infra.dapo_fingerprint import (
        runtime_signature,
    )
    from ksllm4rec_dapo_anchor_multitask_v1_1._infra.grpo_trie import SidPrefixTrie
    from ksllm4rec_dapo_anchor_multitask_v1_1.config import validate_config
    from ksllm4rec_dapo_anchor_multitask_v1_1.evaluation import load_fixed_probe
    from ksllm4rec_dapo_anchor_multitask_v1_1.verify import validate_probe64_report

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    report = _mapping(
        json.loads(args.report.read_text(encoding="utf-8")), "Probe report"
    )
    expected_signature = runtime_signature(config, config_path=args.config)
    if report.get("runtime_signature_sha256") != expected_signature["sha256"]:
        raise RuntimeError("Probe report runtime signature differs from current inputs.")
    if report.get("arm") != config["experiments"]["active"]["arm"]:
        raise RuntimeError("Probe report belongs to another experiment arm.")

    probe_path = Path(config["evaluation"]["fixed_probe"]).resolve()
    fixed = _mapping(report.get("fixed_probe"), "fixed_probe")
    expected_probe = {
        "path": str(probe_path),
        "rows": 1_024,
        "sha256": _sha256(probe_path),
    }
    if fixed != expected_probe:
        raise RuntimeError("Probe report fixed-probe identity differs from current inputs.")
    expected_rows = load_fixed_probe(probe_path)

    trie_dir = Path(config["output"]["trie_dir"]).resolve()
    trie_manifest = trie_dir / "manifest.json"
    if report.get("trie_manifest_sha256") != _sha256(trie_manifest):
        raise RuntimeError("Probe report trie identity differs from current inputs.")
    trie = SidPrefixTrie.load(
        trie_dir, expected_leaf_count=int(config["trie"]["unique_sids"])
    )

    policy = _mapping(report.get("policy_adapter"), "policy_adapter")
    policy_path = Path(str(policy.get("path"))).resolve()
    expected_policy = {
        "path": str(policy_path),
        "adapter_config_sha256": _sha256(policy_path / "adapter_config.json"),
        "adapter_model_sha256": _sha256(policy_path / "adapter_model.safetensors"),
    }
    if policy != expected_policy:
        raise RuntimeError("Probe report policy adapter bytes differ from current files.")

    count = validate_probe64_report(
        report,
        expected_rows=expected_rows,
        trie=trie,
        require_complete=args.require_complete,
    )
    print(
        json.dumps(
            {
                "passed": True,
                "complete": report["complete"],
                "results": count,
                "pass_count": report["pass_count"],
                "policy_adapter": str(policy_path),
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
