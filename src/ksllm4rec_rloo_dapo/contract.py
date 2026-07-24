"""Frozen identities and shared constants for approved RLOO-DAPO profiles."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType

from ksllm4rec_rloo import contract as legacy


SCHEMA_VERSION = 1
SPEC_VERSION = "RLOO-DAPO-T1.2-KV-v2"
PROFILE = "frontier_sft_epoch2_lora64_rloo_dapo_t12_k1_kvcache"


@dataclass(frozen=True)
class FrozenProfile:
    """Content-addressed model identity and isolated outputs for one run."""

    name: str
    spec_version: str
    sft_adapter: Path
    tokenizer: Path
    sft_adapter_sha256: str
    sft_config_sha256: str
    tokenizer_sha256: str
    tokenizer_config_sha256: str
    config_path: Path
    pilot_dir: Path
    run_dir: Path
    log_dir: Path


PROJECT_ROOT = legacy.PROJECT_ROOT
BASE_MODEL = legacy.BASE_MODEL
SOURCE_DATA = legacy.SOURCE_DATA
PROVENANCE = legacy.PROVENANCE
GROUPS_DIR = legacy.GROUPS_DIR
TRIE_DIR = legacy.TRIE_DIR
FIXED_PROBE = legacy.FIXED_PROBE

BASE_MODEL_SHA256 = legacy.BASE_MODEL_SHA256
SOURCE_DATA_SHA256 = legacy.SOURCE_DATA_SHA256
PROVENANCE_SHA256 = legacy.PROVENANCE_SHA256
GROUPS_SHA256 = legacy.GROUPS_SHA256
TRIE_MANIFEST_SHA256 = legacy.TRIE_MANIFEST_SHA256
FIXED_PROBE_SHA256 = legacy.FIXED_PROBE_SHA256

SFT372_PROFILE = "frontier_sft_lora64_lr1p5em4_wd1em3_step372_rloo_dapo_t12_k1_kvcache"
SFT372_ADAPTER = (
    PROJECT_ROOT / "artifacts/sft/platform_exports/"
    "frontier_LORA_6464_000015_正则0001/checkpoint-372/"
    "train-task-tw1g09-1784715567-epoch2"
)

FROZEN_PROFILES = MappingProxyType(
    {
        PROFILE: FrozenProfile(
            name=PROFILE,
            spec_version=SPEC_VERSION,
            sft_adapter=legacy.SFT_ADAPTER,
            tokenizer=legacy.TOKENIZER,
            sft_adapter_sha256=legacy.SFT_ADAPTER_SHA256,
            sft_config_sha256=legacy.SFT_CONFIG_SHA256,
            tokenizer_sha256=legacy.TOKENIZER_SHA256,
            tokenizer_config_sha256=(
                "f321bf488f62da936ab3b32ad919db7cbe4b4489d9e334d6d808df0ca61d8fd8"
            ),
            config_path=(
                PROJECT_ROOT / "configs/rloo/"
                "frontier_sft_epoch2_lora64_g16_rloo_dapo_t12_k1_kvcache.yaml"
            ),
            pilot_dir=(
                PROJECT_ROOT / "artifacts/rloo/pilots/"
                "frontier_lora64_g16_t12_k1_kvcache_v2"
            ),
            run_dir=(
                PROJECT_ROOT / "artifacts/rloo/runs/"
                "frontier_sft_epoch2_lora64_rloo_dapo_t12_k1_kvcache_2epoch"
            ),
            log_dir=(
                PROJECT_ROOT / "operation_logs/rloo/"
                "frontier_sft_epoch2_lora64_rloo_dapo_t12_k1_kvcache_v2"
            ),
        ),
        SFT372_PROFILE: FrozenProfile(
            name=SFT372_PROFILE,
            spec_version="RLOO-DAPO-T1.2-KV-v3-SFT372",
            sft_adapter=SFT372_ADAPTER,
            tokenizer=SFT372_ADAPTER,
            sft_adapter_sha256=(
                "699826c7a276b463f7fe8195a0ee6297083d109a6afe23f161f0985b65684ecb"
            ),
            sft_config_sha256=(
                "55969781f9f855ca35ad5133cb5d0db50f6f8466ce7b078603f26258523e5363"
            ),
            tokenizer_sha256=(
                "cd4d15f596979aecbc11ba12668cb21c9fb8452ce1235cd9d7878cc950170421"
            ),
            tokenizer_config_sha256=(
                "0fde6acb74ecabcd1976fc0ea878a5f3e77b72fd51db3c7a6ffce25f570491f0"
            ),
            config_path=(
                PROJECT_ROOT / "configs/rloo/"
                "frontier_sft_lora64_lr1p5em4_wd1em3_step372_"
                "g16_rloo_dapo_t12_k1_kvcache.yaml"
            ),
            pilot_dir=(
                PROJECT_ROOT / "artifacts/rloo/pilots/"
                "frontier_sft_lora64_lr1p5em4_wd1em3_step372_"
                "g16_t12_k1_kvcache_v3"
            ),
            run_dir=(
                PROJECT_ROOT / "artifacts/rloo/runs/"
                "frontier_sft_lora64_lr1p5em4_wd1em3_step372_"
                "rloo_dapo_t12_k1_kvcache_2effective_epochs"
            ),
            log_dir=(
                PROJECT_ROOT / "operation_logs/rloo/"
                "frontier_sft_lora64_lr1p5em4_wd1em3_step372_"
                "rloo_dapo_t12_k1_kvcache_v3"
            ),
        ),
    }
)


def frozen_profile(name: str) -> FrozenProfile:
    """Return one explicitly approved profile; arbitrary adapters are forbidden."""

    try:
        return FROZEN_PROFILES[name]
    except KeyError as exc:
        raise ValueError(f"Unknown RLOO-DAPO profile: {name!r}.") from exc


# Backward-compatible aliases keep the original profile API stable.
_DEFAULT_PROFILE = frozen_profile(PROFILE)
SFT_ADAPTER = _DEFAULT_PROFILE.sft_adapter
TOKENIZER = _DEFAULT_PROFILE.tokenizer
SFT_ADAPTER_SHA256 = _DEFAULT_PROFILE.sft_adapter_sha256
SFT_CONFIG_SHA256 = _DEFAULT_PROFILE.sft_config_sha256
TOKENIZER_SHA256 = _DEFAULT_PROFILE.tokenizer_sha256
TOKENIZER_CONFIG_SHA256 = _DEFAULT_PROFILE.tokenizer_config_sha256

EXPECTED_SOURCE_ROWS = legacy.EXPECTED_SOURCE_ROWS
EXPECTED_RECOMMEND_ROWS = legacy.EXPECTED_RECOMMEND_ROWS
EXPECTED_RECOMMEND_GROUPS = legacy.EXPECTED_RECOMMEND_GROUPS
EXPECTED_POSITIVE_EDGES = legacy.EXPECTED_POSITIVE_EDGES
EXPECTED_UNIQUE_POSITIVE_SIDS = legacy.EXPECTED_UNIQUE_POSITIVE_SIDS
EXPECTED_TRIE_LEAVES = legacy.EXPECTED_TRIE_LEAVES
EXPECTED_DOMAIN_A_NODES = legacy.EXPECTED_DOMAIN_A_NODES
EXPECTED_DOMAIN_AB_NODES = legacy.EXPECTED_DOMAIN_AB_NODES

POLICY_ADAPTER = legacy.POLICY_ADAPTER
LORA_RANK = legacy.LORA_RANK
LORA_ALPHA = legacy.LORA_ALPHA
LORA_DROPOUT = legacy.LORA_DROPOUT
LORA_TARGET_MODULES = legacy.LORA_TARGET_MODULES
EXPECTED_LORA_TENSOR_COUNT = 392
EXPECTED_LORA_PARAMETER_COUNT = 40_370_176

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

REWARD_VALUES = MappingProxyType(
    {
        "exact": 1.0,
        "same_ab": 0.40,
        "same_a": 0.15,
        "same_domain": 0.01,
        "other_domain": 0.0,
    }
)

EFFECTIVE_EPOCHS = 2
WINDOWS_PER_EFFECTIVE_EPOCH = 532
TOTAL_WINDOWS = EFFECTIVE_EPOCHS * WINDOWS_PER_EFFECTIVE_EPOCH
TOTAL_OPTIMIZER_UPDATES = TOTAL_WINDOWS * OPTIMIZER_UPDATES_PER_WINDOW
LEARNING_RATE = 1.0e-6
WARMUP_WINDOWS = 10
MAX_GRAD_NORM = 1.0

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

RUN_DIR = _DEFAULT_PROFILE.run_dir
LOG_DIR = _DEFAULT_PROFILE.log_dir
