"""Frozen identities and hyperparameters for the approved RLOO Spec V2.0."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType


SPEC_VERSION = "2.0"
SCHEMA_VERSION = 1
PROFILE = "frontier_sft_epoch2_lora64_ref_free_g16_rloo_anchor"

PROJECT_ROOT = Path("/home/lyc/REC_PROJECTS/KSLLM4REC")
BASE_MODEL = Path("/home/lyc/models/OneReason-0.8B-pretrain-competition")
SFT_ADAPTER = (
    PROJECT_ROOT
    / "artifacts/sft/platform_exports/"
    "frontier_feedbackcore_listwise_invariant_epoch2_lora64_20260722/extracted"
)
TOKENIZER = SFT_ADAPTER
SOURCE_DATA = Path(
    "/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/train.jsonl"
)
PROVENANCE = Path(
    "/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/provenance.jsonl"
)
FIXED_PROBE = (
    PROJECT_ROOT / "artifacts/grpo/data/frontier_probe_v1/fixed_probe_1024.jsonl"
)
GROUPS_DIR = PROJECT_ROOT / "artifacts/grpo/data/recommend_groups_frontier_v1"
TRIE_DIR = (
    PROJECT_ROOT
    / "artifacts/grpo/catalog/frontier_all_system_prompt_response_sids_v1"
)
RUN_DIR = (
    PROJECT_ROOT
    / "artifacts/rloo/runs/"
    "frontier_sft_epoch2_lora64_ref_free_g16_rloo_anchor_2epoch"
)
LOG_DIR = (
    PROJECT_ROOT / "operation_logs/rloo/frontier_sft_epoch2_lora64_g16_v1"
)
CALIBRATION_IDS = (
    PROJECT_ROOT / "configs/rloo/frontier_sft_epoch2_lora64_calibration_ids.json"
)

BASE_MODEL_SHA256 = (
    "28e66d2ec528473d335ede2b3faa08eddc53eb8ec93747a449e5e7ec812ede90"
)
SFT_ADAPTER_SHA256 = (
    "c31f1e67cd46b02a27a1ed44136eccb1f83b7055f247c84d4a6924cafa4c1d70"
)
SFT_CONFIG_SHA256 = (
    "041e3f79eaa6acfa181612838c03e275f0fa7883e0b286c82fbb6f164714731e"
)
TOKENIZER_SHA256 = (
    "cd4d15f596979aecbc11ba12668cb21c9fb8452ce1235cd9d7878cc950170421"
)
SOURCE_DATA_SHA256 = (
    "9e3465cedbdaf784ab4720699649c76c256414e2c2089efcd7657f63bae7449a"
)
PROVENANCE_SHA256 = (
    "e56dddff865780273573e06e5a44d4c1dd9a27013c78c166aa6565f7b9ad8908"
)
GROUPS_SHA256 = (
    "a2571d97e4ac231d61d1f468bc23b8e65bebfd5ad7b1a2dd566f1bfd86a9658e"
)
TRIE_MANIFEST_SHA256 = (
    "f51414b13a8982605ebe1f7fec75929245db60c854e3860d3b6c1625a3f33e4a"
)
FIXED_PROBE_SHA256 = (
    "86e050a00506efbaef0e49997ed138462aaddc322eba9fc66227a25654a75e5e"
)

EXPECTED_SOURCE_ROWS = 63_700
EXPECTED_RECOMMEND_ROWS = 30_902
EXPECTED_RECOMMEND_GROUPS = 17_016
EXPECTED_POSITIVE_EDGES = 30_465
EXPECTED_UNIQUE_POSITIVE_SIDS = 29_414
EXPECTED_TRIE_LEAVES = 905_469
EXPECTED_DOMAIN_A_NODES = 10_744
EXPECTED_DOMAIN_AB_NODES = 429_540

POLICY_ADAPTER = "default"
LORA_RANK = 64
LORA_ALPHA = 64
LORA_DROPOUT = 0.0
LORA_TARGET_MODULES = frozenset(
    {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
)

GROUP_SIZE = 16
CHUNK_SIZE = 8
REWARD_VALUES = MappingProxyType(
    {
        "exact": 1.0,
        "same_ab": 0.40,
        "same_a": 0.15,
        "same_domain": 0.01,
        "other_domain": 0.0,
    }
)

EPOCHS = 2
ACCUMULATION_GROUPS = 8
WINDOWS_PER_EPOCH = 2_127
TOTAL_WINDOWS = 4_254
WARMUP_WINDOWS = 128
LEARNING_RATE = 5.0e-6

CALIBRATION_GROUPS = 512
ANCHOR_TARGET_GRADIENT_RATIO = 0.10
ANCHOR_MAX_WEIGHT = 0.05
MIN_GRAD_NORM = 1.0e-12
