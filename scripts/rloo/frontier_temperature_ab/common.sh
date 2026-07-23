#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/home/lyc/REC_PROJECTS/KSLLM4REC"
ENV_PREFIX="/home/lyc/miniconda3/envs/onereason_lora_sft"
PYTHON="${ENV_PREFIX}/bin/python"
CONFIG="${PROJECT_ROOT}/configs/rloo/frontier_epoch2_temperature_t100_t120.yaml"
OUTPUT_ROOT="${PROJECT_ROOT}/operation_logs/rloo/frontier_epoch2_temperature_t100_t120_v1"
SMOKE_ROOT="${PROJECT_ROOT}/operation_logs/rloo/frontier_epoch2_temperature_t100_t120_smoke_v1"
DEVICE="${RLOO_DEVICE:-cuda:0}"

export CONDA_PREFIX="${ENV_PREFIX}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONPATH="${PROJECT_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export TOKENIZERS_PARALLELISM=false
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
