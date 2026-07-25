"""Frozen identities and shared constants for DAPO-Anchor profiles."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType

from ksllm4rec_rloo_dapo import contract as dapo_legacy


SCHEMA_VERSION = 1
SPEC_VERSION = "DAPO-ANCHOR-V1"
PROFILE = "frontier_sft_epoch2_dapo_anchor"


# ── SFT 起点 (复用 RLOO-DAPO 原版) ────────────────────────────────
BASE_MODEL = dapo_legacy.BASE_MODEL
SOURCE_DATA = dapo_legacy.SOURCE_DATA
PROVENANCE = dapo_legacy.PROVENANCE
GROUPS_DIR = dapo_legacy.GROUPS_DIR
TRIE_DIR = dapo_legacy.TRIE_DIR
FIXED_PROBE = dapo_legacy.FIXED_PROBE

BASE_MODEL_SHA256 = dapo_legacy.BASE_MODEL_SHA256
SOURCE_DATA_SHA256 = dapo_legacy.SOURCE_DATA_SHA256
PROVENANCE_SHA256 = dapo_legacy.PROVENANCE_SHA256
GROUPS_SHA256 = dapo_legacy.GROUPS_SHA256
TRIE_MANIFEST_SHA256 = dapo_legacy.TRIE_MANIFEST_SHA256
FIXED_PROBE_SHA256 = dapo_legacy.FIXED_PROBE_SHA256

EXPECTED_SOURCE_ROWS = dapo_legacy.EXPECTED_SOURCE_ROWS
EXPECTED_RECOMMEND_ROWS = dapo_legacy.EXPECTED_RECOMMEND_ROWS
EXPECTED_RECOMMEND_GROUPS = dapo_legacy.EXPECTED_RECOMMEND_GROUPS
EXPECTED_POSITIVE_EDGES = dapo_legacy.EXPECTED_POSITIVE_EDGES
EXPECTED_UNIQUE_POSITIVE_SIDS = dapo_legacy.EXPECTED_UNIQUE_POSITIVE_SIDS
EXPECTED_TRIE_LEAVES = dapo_legacy.EXPECTED_TRIE_LEAVES
EXPECTED_DOMAIN_A_NODES = dapo_legacy.EXPECTED_DOMAIN_A_NODES
EXPECTED_DOMAIN_AB_NODES = dapo_legacy.EXPECTED_DOMAIN_AB_NODES

POLICY_ADAPTER = dapo_legacy.POLICY_ADAPTER
LORA_RANK = dapo_legacy.LORA_RANK
LORA_ALPHA = dapo_legacy.LORA_ALPHA
LORA_DROPOUT = dapo_legacy.LORA_DROPOUT
LORA_TARGET_MODULES = dapo_legacy.LORA_TARGET_MODULES
EXPECTED_LORA_TENSOR_COUNT = dapo_legacy.EXPECTED_LORA_TENSOR_COUNT
EXPECTED_LORA_PARAMETER_COUNT = dapo_legacy.EXPECTED_LORA_PARAMETER_COUNT

SFT372_ADAPTER = (
    dapo_legacy.PROJECT_ROOT / "artifacts/sft/platform_exports/"
    "frontier_LORA_6464_000015_正则0001/checkpoint-372/"
    "train-task-tw1g09-1784715567-epoch2"
)

# ── 训练常量 ──────────────────────────────────────────────────────────
GROUP_SIZE = 16
PROMPT_BATCH_SIZE = 8
MAX_ACTIVE_SEQUENCES = 16
CANDIDATES_PER_PROMPT_WAVE = 2
LOSS_CHUNK_SIZE = 8

EFFECTIVE_GROUPS_PER_WINDOW = 32
MINIBATCH_GROUPS = 8
OPTIMIZER_UPDATES_PER_WINDOW = 4
NUM_ITERATIONS = 1

TEMPERATURE = 1.2
CLIP_RATIO_LOW = 0.8
CLIP_RATIO_HIGH = 1.28
SAMPLE_CANONICAL_MAX_LOGP_DIFF = 5.0e-3
CANONICAL_REPLAY_MAX_LOGP_DIFF = 1.0e-5

# ── Anchor 常量 ──────────────────────────────────────────────────────
ANCHOR_WEIGHT = 0.05          # 固定小权重, RL:anchor ≈ 20:1
ANCHOR_BATCH_SIZE = 16        # anchor phase 每 optimizer.step 的组数
ANCHOR_WARMUP_WINDOWS = 10    # warmup 期间不参与,等 RL 梯度建立

# ── Reward 五档保持不变 ──────────────────────────────────────────────
REWARD_VALUES = MappingProxyType(
    {
        "exact": 1.0,
        "same_ab": 0.40,
        "same_a": 0.15,
        "same_domain": 0.01,
        "other_domain": 0.0,
    }
)

# ── Epoch 定义 ───────────────────────────────────────────────────────
EFFECTIVE_EPOCHS = 2
WINDOWS_PER_EFFECTIVE_EPOCH = 532
TOTAL_WINDOWS = EFFECTIVE_EPOCHS * WINDOWS_PER_EFFECTIVE_EPOCH  # 1064
TOTAL_OPTIMIZER_UPDATES = TOTAL_WINDOWS * OPTIMIZER_UPDATES_PER_WINDOW  # 4256
WARMUP_WINDOWS = 10
LEARNING_RATE = 1.0e-6
MAX_GRAD_NORM = 1.0

# ── KV Cache / 显存 ──────────────────────────────────────────────────
KV_CACHE_BUDGET_GIB = 12.0
MAX_RESERVED_GIB = 20.0
MODEL_NUM_LAYERS = 28
MODEL_NUM_KV_HEADS = 8
MODEL_HEAD_DIM = 128
MODEL_CACHE_DTYPE_BYTES = 2
MAX_COMPLETION_LENGTH = 32
MAX_OBSERVED_PROMPT_LENGTH = 2_929
MAX_OPTIMIZED_WINDOW_SECONDS = 600.0

EXECUTION_DEVICE = "cuda:0"
EXECUTION_CUDA_VISIBLE_DEVICES = "0"
EXPECTED_GPU_IDENTITY = MappingProxyType(
    {
        "device": EXECUTION_DEVICE,
        "name": "NVIDIA GeForce RTX 4090",
        "uuid": "e1f24bb9-9bcb-c31f-52ba-fb5f63a146d5",
        "total_memory": 25_756_696_576,
        "compute_capability": [8, 9],
    }
)
