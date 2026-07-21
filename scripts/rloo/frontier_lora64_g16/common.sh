#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/lyc/REC_PROJECTS/KSLLM4REC"
ENV_PREFIX="/home/lyc/miniconda3/envs/onereason_lora_sft"
PYTHON="${ENV_PREFIX}/bin/python"
CONFIG="${PROJECT_ROOT}/configs/rloo/frontier_sft_epoch2_lora64_g16.yaml"
CALIBRATION_IDS="${PROJECT_ROOT}/configs/rloo/frontier_sft_epoch2_lora64_calibration_ids.json"
GROUPS_FILE="${PROJECT_ROOT}/artifacts/grpo/data/recommend_groups_frontier_v1/groups.jsonl"
TRIE_DIR="${PROJECT_ROOT}/artifacts/grpo/catalog/frontier_all_system_prompt_response_sids_v1"
INITIAL_ADAPTER="${PROJECT_ROOT}/artifacts/sft/platform_exports/frontier_feedbackcore_listwise_invariant_epoch2_lora64_20260722/extracted"
RUN_DIR="${PROJECT_ROOT}/artifacts/rloo/runs/frontier_sft_epoch2_lora64_ref_free_g16_rloo_anchor_2epoch"
LOG_ROOT="${PROJECT_ROOT}/operation_logs/rloo/frontier_sft_epoch2_lora64_g16_v1"
DEVICE="${RLOO_DEVICE:-cuda:0}"

export CONDA_PREFIX="${ENV_PREFIX}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
