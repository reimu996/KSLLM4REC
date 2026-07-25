"""Frozen identities and shared constants for DAPO-Anchor profiles.

Every constant here is a *literal* — no forwarding through `_infra.dapo_contract`
or any sibling package. This is the top-level contract seen by trainer.py /
objective.py / verify.py / config.py of the DAPO-Anchor method.

Values were captured at vendor commit 213a1002. Bit-for-bit identical to the
pre-vendor state in which these constants were transitively forwarded through
`ksllm4rec_rloo_dapo.contract → ksllm4rec_rloo.contract`.

Cross-check (must all still hold):
    from ksllm4rec_dapo_anchor import contract as c
    from ksllm4rec_dapo_anchor._infra import dapo_contract as legacy
    assert c.BASE_MODEL_SHA256 == legacy.BASE_MODEL_SHA256   # etc.
"""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType


SCHEMA_VERSION = 1
SPEC_VERSION = "DAPO-ANCHOR-V1"
PROFILE = "frontier_sft_epoch2_dapo_anchor"

PROJECT_ROOT = Path("/home/lyc/REC_PROJECTS/KSLLM4REC")


# ── SFT 起点 (与 RLOO-DAPO 原版数值等同, 已落地为字面量) ─────────────
BASE_MODEL = Path("/home/lyc/models/OneReason-0.8B-pretrain-competition")
SOURCE_DATA = Path(
    "/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/train.jsonl"
)
PROVENANCE = Path(
    "/home/lyc/Data/frontier_feedbackcore_listwise_invariant_v1/provenance.jsonl"
)
GROUPS_DIR = (
    PROJECT_ROOT / "artifacts/grpo/data/recommend_groups_frontier_v1"
)
TRIE_DIR = (
    PROJECT_ROOT
    / "artifacts/grpo/catalog/frontier_all_system_prompt_response_sids_v1"
)
FIXED_PROBE = (
    PROJECT_ROOT / "artifacts/grpo/data/frontier_probe_v1/fixed_probe_1024.jsonl"
)

BASE_MODEL_SHA256 = "28e66d2ec528473d335ede2b3faa08eddc53eb8ec93747a449e5e7ec812ede90"
SOURCE_DATA_SHA256 = "9e3465cedbdaf784ab4720699649c76c256414e2c2089efcd7657f63bae7449a"
PROVENANCE_SHA256 = "e56dddff865780273573e06e5a44d4c1dd9a27013c78c166aa6565f7b9ad8908"
GROUPS_SHA256 = "a2571d97e4ac231d61d1f468bc23b8e65bebfd5ad7b1a2dd566f1bfd86a9658e"
TRIE_MANIFEST_SHA256 = "f51414b13a8982605ebe1f7fec75929245db60c854e3860d3b6c1625a3f33e4a"
FIXED_PROBE_SHA256 = "86e050a00506efbaef0e49997ed138462aaddc322eba9fc66227a25654a75e5e"

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
    {"q_proj", "k_proj", "v_proj", "o_proj",
     "gate_proj", "up_proj", "down_proj"}
)
EXPECTED_LORA_TENSOR_COUNT = 392
EXPECTED_LORA_PARAMETER_COUNT = 40_370_176

SFT372_ADAPTER = (
    PROJECT_ROOT / "artifacts/sft/platform_exports/"
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
