#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
readonly ROOT

# Only PYTHON is intentionally overridable. Every value that identifies this
# training profile is frozen here so an inherited environment cannot redirect
# training to another config, output directory, or GPU.
readonly PYTHON="${PYTHON:-/home/lyc/miniconda3/envs/onereason_lora_sft/bin/python}"
readonly CONFIG="${ROOT}/configs/rloo/frontier_sft372_g16_dapo_anchor_v2_2.yaml"
readonly LOG_ROOT="${ROOT}/operation_logs/rloo/dapo_anchor_sft372_v2_2"
readonly RUN_DIR="${ROOT}/artifacts/rloo/runs/dapo_anchor_sft372_v2_2_2effective_epochs"
readonly PILOT_DIR="${ROOT}/artifacts/rloo/pilots/dapo_anchor_sft372_v2_2_w0"
readonly TRAIN_LOG="${LOG_ROOT}/full_train.stdout.log"
readonly LAUNCHER="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/train.py"
readonly CALIBRATION_LAUNCHER="$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/calibrate.py"
readonly CALIBRATION_REPORT="${LOG_ROOT}/anchor_calibration.json"
readonly DEVICE="cuda:0"
export CUDA_VISIBLE_DEVICES="0"
readonly CUDA_VISIBLE_DEVICES

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
