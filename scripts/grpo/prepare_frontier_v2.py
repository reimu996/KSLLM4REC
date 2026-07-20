#!/usr/bin/env python
"""Build the Frontier-specific GRPO data, SID trie, and fixed probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ksllm4rec_grpo.frontier_data import (
    build_frontier_groups,
    build_frontier_probe,
    build_frontier_trie,
    write_source_lock_report,
)


PROJECT_ROOT = Path("/home/lyc/REC_PROJECTS/KSLLM4REC")
DEFAULT_SOURCE = Path(
    "/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/train.jsonl"
)
DEFAULT_PROVENANCE = Path(
    "/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/provenance.jsonl"
)
DEFAULT_SOURCE_LOCK = PROJECT_ROOT / "artifacts/grpo/data/frontier_source_lock_v1"
DEFAULT_GROUPS = PROJECT_ROOT / "artifacts/grpo/data/recommend_groups_frontier_v1"
DEFAULT_TRIE = PROJECT_ROOT / (
    "artifacts/grpo/catalog/frontier_all_system_prompt_response_sids_v1"
)
DEFAULT_PROBE = PROJECT_ROOT / "artifacts/grpo/data/frontier_probe_v1"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--provenance", type=Path, default=DEFAULT_PROVENANCE)
    parser.add_argument("--source-lock-dir", type=Path, default=DEFAULT_SOURCE_LOCK)
    parser.add_argument("--groups-dir", type=Path, default=DEFAULT_GROUPS)
    parser.add_argument("--trie-dir", type=Path, default=DEFAULT_TRIE)
    parser.add_argument("--probe-dir", type=Path, default=DEFAULT_PROBE)
    parser.add_argument("--allow-contract-drift", action="store_true")
    return parser


def main() -> int:
    args = _parser().parse_args()
    source = args.source.resolve()
    provenance = args.provenance.resolve()
    enforce = not args.allow_contract_drift
    args.source_lock_dir.mkdir(parents=True, exist_ok=True)
    lock = write_source_lock_report(
        source,
        provenance,
        args.source_lock_dir / "source_manifest.json",
    )
    groups = build_frontier_groups(
        source,
        provenance,
        args.groups_dir,
        enforce_contract=enforce,
    )
    trie = build_frontier_trie(
        source,
        provenance,
        args.trie_dir,
        enforce_contract=enforce,
    )
    probe = build_frontier_probe(
        source,
        provenance,
        args.probe_dir / "fixed_probe_1024.jsonl",
        enforce_contract=enforce,
    )
    print(
        json.dumps(
            {
                "source_lock": lock,
                "groups": groups,
                "trie": trie,
                "probe": probe,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
