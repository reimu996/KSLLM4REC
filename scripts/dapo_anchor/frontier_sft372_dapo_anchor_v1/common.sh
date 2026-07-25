#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
readonly ROOT

# Only PYTHON is intentionally overridable. Every value that identifies this
# training profile is frozen here so an inherited environment cannot redirect
# training to another config, output directory, or GPU.
readonly PYTHON="${PYTHON:-/home/lyc/miniconda3/envs/onereason_lora_sft/bin/python}"
readonly CONFIG="${ROOT}/configs/rloo/frontier_sft_lora64_lr1p5em4_wd1em3_step372_g16_dapo_anchor005_t12_k1_kvcache.yaml"
readonly LOG_ROOT="${ROOT}/operation_logs/rloo/dapo_anchor_sft372"
readonly RUN_DIR="${ROOT}/artifacts/rloo/runs/dapo_anchor_sft372_2effective_epochs"
readonly TRAIN_LOG="${LOG_ROOT}/full_train.stdout.log"
readonly LAUNCHER="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/train.py"
readonly DEVICE="cuda:0"
export CUDA_VISIBLE_DEVICES="0"
readonly CUDA_VISIBLE_DEVICES

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
