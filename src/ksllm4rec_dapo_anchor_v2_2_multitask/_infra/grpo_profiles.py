"""Immutable identities for the approved GRPO experiment profiles.

The original V3.1 run is intentionally kept as the default profile.  The
Frontier profile changes only the frozen inputs and derived-data locations;
algorithmic values remain in the YAML contract.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Mapping


PROJECT_ROOT = Path("/home/lyc/REC_PROJECTS/KSLLM4REC")
BASE_MODEL = Path("/home/lyc/models/OneReason-0.8B-pretrain-competition")
DEFAULT_PROFILE_NAME = "baseline_v3_1"
FRONTIER_PROFILE_NAME = "frontier_sft_epoch2_v2"
DAPO_ANCHOR_PROFILE_NAME = "frontier_sft_epoch2_dapo_anchor"


@dataclass(frozen=True)
class GRPOProfile:
    """Frozen paths, hashes, and data counts for one GRPO experiment."""

    name: str
    spec_version: str
    schema_version: int
    base_model: Path
    sft_adapter: Path
    tokenizer: Path
    source: Path
    provenance: Path | None
    fixed_probe: Path | None
    groups_dir: Path
    trie_dir: Path
    run_dir: Path
    log_dir: Path
    source_sha256: str
    provenance_sha256: str | None
    adapter_sha256: str
    adapter_config_sha256: str
    groups_sha256: str | None
    trie_manifest_sha256: str | None
    fixed_probe_sha256: str | None
    source_rows: int
    recommend_rows: int
    groups: int
    positives: int
    unique_positive_sids: int
    unique_sids: int
    domain_a_nodes: int
    domain_ab_nodes: int
    domain_sids: Mapping[str, int]
    trie_strategy: str
    probe_reachable: Mapping[str, int] | None = None

    @property
    def source_record(self) -> Path:
        return self.source

    def optimizer_steps(self, *, epochs: int, accumulation_groups: int) -> int:
        """Return the exact number of optimizer windows, including a tail."""

        if epochs < 1 or accumulation_groups < 1:
            raise ValueError("epochs and accumulation_groups must be positive")
        return ((self.groups + accumulation_groups - 1) // accumulation_groups) * epochs


BASELINE_PROFILE = GRPOProfile(
    name=DEFAULT_PROFILE_NAME,
    spec_version="3.1",
    schema_version=3,
    base_model=BASE_MODEL,
    sft_adapter=PROJECT_ROOT / "artifacts/sft/runs/full_epoch_001",
    tokenizer=PROJECT_ROOT / "artifacts/sft/runs/full_epoch_001",
    source=Path("/home/lyc/Data/HF_kuaishou-llmrec-sft-baseline-0.91/train.jsonl"),
    provenance=None,
    fixed_probe=PROJECT_ROOT / "artifacts/orpo/data/all_tasks_32705/fixed_probe_1024.jsonl",
    groups_dir=PROJECT_ROOT / "artifacts/grpo/data/recommend_groups_v3_1",
    trie_dir=PROJECT_ROOT / "artifacts/grpo/catalog/baseline_all_sids_v3_1",
    run_dir=PROJECT_ROOT / "artifacts/grpo/runs/sft_grpo_g8_forcedgt_p050_zscore_2epoch",
    log_dir=PROJECT_ROOT / "operation_logs/grpo/v3_1",
    source_sha256="4d6b29d76974c9a1517c1b583858e744cae019cb26e1e2d90066e237ebbcf5f8",
    provenance_sha256=None,
    adapter_sha256="77ac75d6ad558bdafc97cfb096f9608c59f4382c2884f17dceaf0fd943e0b710",
    adapter_config_sha256="5128629f9fe624a5807a8e8e728d3a8763936a01f8c9bc9df1ce4b9d00a9c5ec",
    groups_sha256="1c85ad933db46284d1417b7c2e83e31c2d379997a4e966439435f04ca75a243f",
    trie_manifest_sha256="737b2f69a568ba21d707f88b9fbd52a2d3c53ebb1b3aab6f2478e87530dbc066",
    fixed_probe_sha256="8b8c64bcf9f4590dc006f900ded27d7008816d56ea1b3df6ccfd55a4e63b5fb1",
    source_rows=32_705,
    recommend_rows=18_651,
    groups=6_378,
    positives=18_651,
    unique_positive_sids=18_408,
    unique_sids=768_593,
    domain_a_nodes=10_198,
    domain_ab_nodes=385_077,
    domain_sids=MappingProxyType(
        {"ad": 115_399, "living": 28_441, "prod": 294_325, "video": 330_428}
    ),
    trie_strategy="baseline_all_system_prompt_response_sids",
    probe_reachable=MappingProxyType({"text_to_sid": 256, "recommend": 111}),
)


FRONTIER_PROFILE = GRPOProfile(
    name=FRONTIER_PROFILE_NAME,
    spec_version="2.0",
    schema_version=4,
    base_model=BASE_MODEL,
    sft_adapter=PROJECT_ROOT
    / "artifacts/sft/platform_exports/frontier_feedbackcore_listwise_invariant_epoch2_20260719/extracted",
    tokenizer=PROJECT_ROOT
    / "artifacts/sft/platform_exports/frontier_feedbackcore_listwise_invariant_epoch2_20260719/extracted",
    source=Path("/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/train.jsonl"),
    provenance=Path(
        "/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/provenance.jsonl"
    ),
    fixed_probe=PROJECT_ROOT / "artifacts/grpo/data/frontier_probe_v1/fixed_probe_1024.jsonl",
    groups_dir=PROJECT_ROOT / "artifacts/grpo/data/recommend_groups_frontier_v1",
    trie_dir=PROJECT_ROOT
    / "artifacts/grpo/catalog/frontier_all_system_prompt_response_sids_v1",
    run_dir=PROJECT_ROOT
    / "artifacts/grpo/runs/frontier_sft_epoch2_grpo_g8_forcedgt_p050_zscore_2epoch",
    log_dir=PROJECT_ROOT / "operation_logs/grpo/frontier_sft_epoch2_v2",
    source_sha256="9e3465cedbdaf784ab4720699649c76c256414e2c2089efcd7657f63bae7449a",
    provenance_sha256="e56dddff865780273573e06e5a44d4c1dd9a27013c78c166aa6565f7b9ad8908",
    adapter_sha256="b7a50ca748f83059ddf27c5a8053fda3095a541619ee67a27460f9535b865f63",
    adapter_config_sha256="3c64591b20398ec00baee501ed71e1146773a509c140d7dbbc95e1100e76c8cf",
    groups_sha256="a2571d97e4ac231d61d1f468bc23b8e65bebfd5ad7b1a2dd566f1bfd86a9658e",
    trie_manifest_sha256="f51414b13a8982605ebe1f7fec75929245db60c854e3860d3b6c1625a3f33e4a",
    fixed_probe_sha256="86e050a00506efbaef0e49997ed138462aaddc322eba9fc66227a25654a75e5e",
    source_rows=63_700,
    recommend_rows=30_902,
    groups=17_016,
    positives=30_465,
    unique_positive_sids=29_414,
    unique_sids=905_469,
    domain_a_nodes=10_744,
    domain_ab_nodes=429_540,
    domain_sids=MappingProxyType(
        {"ad": 121_769, "living": 35_907, "prod": 309_414, "video": 438_379}
    ),
    trie_strategy="frontier_all_system_prompt_response_sids",
    probe_reachable=MappingProxyType({"text_to_sid": 512, "recommend": 512}),
)


DAPO_ANCHOR_PROFILE = GRPOProfile(
    name=DAPO_ANCHOR_PROFILE_NAME,
    spec_version="2.0",
    schema_version=4,
    base_model=BASE_MODEL,
    sft_adapter=PROJECT_ROOT
    / "artifacts/sft/platform_exports/"
    "frontier_LORA_6464_000015_正则0001/checkpoint-372/"
    "train-task-tw1g09-1784715567-epoch2",
    tokenizer=PROJECT_ROOT
    / "artifacts/sft/platform_exports/"
    "frontier_LORA_6464_000015_正则0001/checkpoint-372/"
    "train-task-tw1g09-1784715567-epoch2",
    source=Path("/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/train.jsonl"),
    provenance=Path(
        "/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/provenance.jsonl"
    ),
    fixed_probe=PROJECT_ROOT / "artifacts/grpo/data/frontier_probe_v1/fixed_probe_1024.jsonl",
    groups_dir=PROJECT_ROOT / "artifacts/grpo/data/recommend_groups_frontier_v1",
    trie_dir=PROJECT_ROOT
    / "artifacts/grpo/catalog/frontier_all_system_prompt_response_sids_v1",
    run_dir=PROJECT_ROOT
    / "artifacts/rloo/runs/dapo_anchor_sft372_2effective_epochs",
    log_dir=PROJECT_ROOT / "operation_logs/rloo/dapo_anchor_sft372",
    source_sha256="9e3465cedbdaf784ab4720699649c76c256414e2c2089efcd7657f63bae7449a",
    provenance_sha256="e56dddff865780273573e06e5a44d4c1dd9a27013c78c166aa6565f7b9ad8908",
    adapter_sha256="699826c7a276b463f7fe8195a0ee6297083d109a6afe23f161f0985b65684ecb",
    adapter_config_sha256="55969781f9f855ca35ad5133cb5d0db50f6f8466ce7b078603f26258523e5363",
    groups_sha256="a2571d97e4ac231d61d1f468bc23b8e65bebfd5ad7b1a2dd566f1bfd86a9658e",
    trie_manifest_sha256="f51414b13a8982605ebe1f7fec75929245db60c854e3860d3b6c1625a3f33e4a",
    fixed_probe_sha256="86e050a00506efbaef0e49997ed138462aaddc322eba9fc66227a25654a75e5e",
    source_rows=63_700,
    recommend_rows=30_902,
    groups=17_016,
    positives=30_465,
    unique_positive_sids=29_414,
    unique_sids=905_469,
    domain_a_nodes=10_744,
    domain_ab_nodes=429_540,
    domain_sids=MappingProxyType(
        {"ad": 121_769, "living": 35_907, "prod": 309_414, "video": 438_379}
    ),
    trie_strategy="frontier_all_system_prompt_response_sids",
    probe_reachable=MappingProxyType({"text_to_sid": 512, "recommend": 512}),
)


PROFILES = MappingProxyType(
    {
        BASELINE_PROFILE.name: BASELINE_PROFILE,
        FRONTIER_PROFILE.name: FRONTIER_PROFILE,
        DAPO_ANCHOR_PROFILE.name: DAPO_ANCHOR_PROFILE,
    }
)
PROFILE_NAMES = tuple(PROFILES)


def get_profile(profile: str | GRPOProfile = DEFAULT_PROFILE_NAME) -> GRPOProfile:
    if isinstance(profile, GRPOProfile):
        return profile
    try:
        return PROFILES[profile]
    except KeyError as exc:
        raise ValueError(
            f"Unknown GRPO profile {profile!r}; expected one of {PROFILE_NAMES!r}."
        ) from exc
