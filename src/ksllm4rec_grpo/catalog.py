"""Build the V3.1 baseline-only SID prefix tree."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .contract import (
    EXPECTED_BASELINE_UNIQUE_SIDS,
    EXPECTED_DOMAIN_A_NODES,
    EXPECTED_DOMAIN_AB_NODES,
    SOURCE_DATA_SHA256,
)
from .data import scan_baseline_sids
from .trie import SidPrefixTrie


def build_baseline_trie(source: Path, output_dir: Path) -> dict[str, Any]:
    """Extract all baseline SIDs, build the compressed trie, and save it."""

    sids, domain_counts = scan_baseline_sids(source)
    trie = SidPrefixTrie.from_sids(sids)
    counts = trie.counts
    if counts["leaves"] != EXPECTED_BASELINE_UNIQUE_SIDS:
        raise RuntimeError("Compressed trie lost baseline SID leaves.")
    if counts["a_nodes"] != EXPECTED_DOMAIN_A_NODES:
        raise RuntimeError(
            f"Expected {EXPECTED_DOMAIN_A_NODES} domain/a nodes, "
            f"got {counts['a_nodes']}."
        )
    if counts["ab_nodes"] != EXPECTED_DOMAIN_AB_NODES:
        raise RuntimeError(
            f"Expected {EXPECTED_DOMAIN_AB_NODES} domain/a/b nodes, "
            f"got {counts['ab_nodes']}."
        )
    metadata = {
        "strategy": "baseline_all_system_prompt_response_sids",
        "source": str(source.resolve()),
        "source_sha256": SOURCE_DATA_SHA256,
        "unique_sids": len(sids),
        "domain_unique_sids": domain_counts,
    }
    return trie.save(output_dir, metadata=metadata)
