"""Frozen constants for the approved OneReason LoRA GRPO Spec V3.1."""

from __future__ import annotations

from pathlib import Path

from .grpo_profiles import (
    BASELINE_PROFILE,
    GRPOProfile,
    get_profile,
)


SPEC_VERSION = "3.1"
SCHEMA_VERSION = 3
PROJECT_ROOT = Path("/home/lyc/REC_PROJECTS/KSLLM4REC")
BASE_MODEL = Path("/home/lyc/models/OneReason-0.8B-pretrain-competition")
SFT_ADAPTER = PROJECT_ROOT / "artifacts/sft/runs/full_epoch_001"
SOURCE_DATA = Path("/home/lyc/Data/HF_kuaishou-llmrec-sft-baseline-0.91/train.jsonl")
FIXED_PROBE = (
    PROJECT_ROOT / "artifacts/orpo/data/all_tasks_32705/fixed_probe_1024.jsonl"
)

GROUPS_DIR = PROJECT_ROOT / "artifacts/grpo/data/recommend_groups_v3_1"
TRIE_DIR = PROJECT_ROOT / "artifacts/grpo/catalog/baseline_all_sids_v3_1"
RUN_DIR = PROJECT_ROOT / "artifacts/grpo/runs/sft_grpo_g8_forcedgt_p050_zscore_2epoch"
LOG_DIR = PROJECT_ROOT / "operation_logs/grpo/v3_1"

BASE_MODEL_SHA256 = "28e66d2ec528473d335ede2b3faa08eddc53eb8ec93747a449e5e7ec812ede90"
SFT_ADAPTER_SHA256 = "77ac75d6ad558bdafc97cfb096f9608c59f4382c2884f17dceaf0fd943e0b710"
SFT_CONFIG_SHA256 = "5128629f9fe624a5807a8e8e728d3a8763936a01f8c9bc9df1ce4b9d00a9c5ec"
SOURCE_DATA_SHA256 = "4d6b29d76974c9a1517c1b583858e744cae019cb26e1e2d90066e237ebbcf5f8"
FIXED_PROBE_SHA256 = "8b8c64bcf9f4590dc006f900ded27d7008816d56ea1b3df6ccfd55a4e63b5fb1"
GROUPS_SHA256 = "1c85ad933db46284d1417b7c2e83e31c2d379997a4e966439435f04ca75a243f"
TRIE_MANIFEST_SHA256 = (
    "737b2f69a568ba21d707f88b9fbd52a2d3c53ebb1b3aab6f2478e87530dbc066"
)

EXPECTED_SOURCE_ROWS = 32_705
EXPECTED_RECOMMEND_ROWS = 18_651
EXPECTED_RECOMMEND_GROUPS = 6_378
EXPECTED_UNIQUE_RECOMMEND_POSITIVES = 18_408
EXPECTED_BASELINE_UNIQUE_SIDS = 768_593
EXPECTED_DOMAIN_SIDS = {
    "ad": 115_399,
    "living": 28_441,
    "prod": 294_325,
    "video": 330_428,
}
EXPECTED_DOMAIN_A_NODES = 10_198
EXPECTED_DOMAIN_AB_NODES = 385_077
EXPECTED_PROBE_REACHABLE = {"text_to_sid": 256, "recommend": 111}
MAX_SID_COMPONENT = 8_191
MODEL_VOCAB_SIZE = 176_253

POSITIVE_SET_SIZE_DISTRIBUTION = {
    1: 2927,
    2: 1217,
    3: 707,
    4: 433,
    5: 281,
    6: 186,
    7: 134,
    8: 100,
    9: 71,
    10: 70,
    11: 38,
    12: 38,
    13: 37,
    14: 17,
    15: 20,
    16: 9,
    17: 14,
    18: 10,
    19: 18,
    20: 34,
    21: 14,
    22: 1,
    23: 2,
}

DOMAINS = ("video", "prod", "ad", "living")
DOMAIN_TO_ID = {domain: index for index, domain in enumerate(DOMAINS)}
ID_TO_DOMAIN = dict(enumerate(DOMAINS))
RESPONSE_PREFIX = {
    "video": "该用户最近喜欢的视频有: ",
    "prod": "该用户最近点击了商品: ",
    "ad": "该用户最近感兴趣的广告有: ",
    "living": "该用户最近首次打赏了主播: ",
}
EMPTY_THINK = "<think>\n</think>\n"

EXPECTED_TOKEN_IDS = {
    "eos": 151_645,
    "think_open": 151_667,
    "think_close": 151_668,
    "a_zero": 151_669,
    "a_last": 159_860,
    "b_zero": 159_861,
    "b_last": 168_052,
    "c_zero": 168_053,
    "c_last": 176_244,
    "video": 176_245,
    "prod": 176_247,
    "living": 176_249,
    "ad": 176_251,
}


def profile_for_config(config: dict | None = None) -> GRPOProfile:
    """Resolve the frozen input contract without changing V3.1 aliases.

    Older callers pass no config and therefore receive the original baseline
    contract.  New callers pass the parsed YAML, whose ``profile`` field
    selects the Frontier contract.
    """

    if config is None:
        return BASELINE_PROFILE
    name = config.get("profile") or config.get("profile_name")
    if name is None:
        # The historical file predates named profiles and is unambiguously V3.1.
        return BASELINE_PROFILE
    return get_profile(str(name))


def expected_trie_leaf_count(config: dict | None = None) -> int:
    return profile_for_config(config).unique_sids


def expected_trie_node_counts(config: dict | None = None) -> tuple[int, int]:
    profile = profile_for_config(config)
    return profile.domain_a_nodes, profile.domain_ab_nodes


def expected_data_counts(config: dict | None = None) -> dict[str, int]:
    profile = profile_for_config(config)
    return {
        "source_rows": profile.source_rows,
        "recommend_rows": profile.recommend_rows,
        "groups": profile.groups,
        "positives": profile.positives,
        "unique_positive_sids": profile.unique_positive_sids,
    }
