#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/lyc/REC_PROJECTS/KSLLM4REC"
ENV_PREFIX="/home/lyc/miniconda3/envs/onereason_lora_sft"
PYTHON="${ENV_PREFIX}/bin/python"
CONFIG="${PROJECT_ROOT}/configs/grpo/frontier_sft_epoch2_grpo.yaml"
DATA_DIR="${PROJECT_ROOT}/artifacts/grpo/data/recommend_groups_frontier_v1"
TRIE_DIR="${PROJECT_ROOT}/artifacts/grpo/catalog/frontier_all_system_prompt_response_sids_v1"
PROBE_DIR="${PROJECT_ROOT}/artifacts/grpo/data/frontier_probe_v1"
RUN_DIR="${PROJECT_ROOT}/artifacts/grpo/runs/frontier_sft_epoch2_grpo_g8_forcedgt_p050_zscore_2epoch"
LOG_ROOT="${PROJECT_ROOT}/operation_logs/grpo/frontier_sft_epoch2_v2"
DEVICE="${GRPO_DEVICE:-cuda:0}"

export CONDA_PREFIX="${ENV_PREFIX}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
