#!/usr/bin/env bash

PROJECT_ROOT="/home/lyc/REC_PROJECTS/KSLLM4REC"
ENV_PREFIX="/home/lyc/miniconda3/envs/onereason_lora_sft"
PYTHON="${ENV_PREFIX}/bin/python"
CONFIG="${PROJECT_ROOT}/configs/grpo/onereason_lora_grpo.yaml"
DATA_DIR="${PROJECT_ROOT}/artifacts/grpo/data/recommend_groups_v3_1"
TRIE_DIR="${PROJECT_ROOT}/artifacts/grpo/catalog/baseline_all_sids_v3_1"
RUN_DIR="${PROJECT_ROOT}/artifacts/grpo/runs/sft_grpo_g8_forcedgt_p050_zscore_2epoch"
LOG_ROOT="${PROJECT_ROOT}/operation_logs/grpo/v3_1"

export CONDA_PREFIX="${ENV_PREFIX}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
