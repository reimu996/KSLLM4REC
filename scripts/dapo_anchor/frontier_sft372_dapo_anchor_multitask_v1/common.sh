#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
readonly SCRIPT_DIR ROOT

readonly PYTHON="${PYTHON:-/home/lyc/miniconda3/envs/onereason_lora_sft/bin/python}"
ARM="${KSLLM4REC_ARM:-C}"
ARM="${ARM^^}"
case "${ARM}" in
  A|B) ARM_SUFFIX="_arm_${ARM,,}" ;;
  C) ARM_SUFFIX="" ;;
  *) echo "KSLLM4REC_ARM must be A, B, or C; got ${ARM}." >&2; return 2 ;;
esac
readonly ARM ARM_SUFFIX
readonly CONFIG="${ROOT}/configs/rloo/frontier_sft372_g16_dapo_anchor_multitask_v1${ARM_SUFFIX}.yaml"
readonly LOG_ROOT="${ROOT}/operation_logs/rloo/dapo_anchor_multitask_sft372_v1${ARM_SUFFIX}"
readonly RUN_DIR="${ROOT}/artifacts/rloo/runs/dapo_anchor_multitask_sft372_v1${ARM_SUFFIX}"
readonly PILOT_DIR="${ROOT}/artifacts/rloo/pilots/dapo_anchor_multitask_sft372_v1${ARM_SUFFIX}_one_window"
readonly TEXT_GROUPS_DIR="${ROOT}/artifacts/grpo/data/text_to_sid_groups_frontier_v1"
readonly CALIBRATION_REPORT="${LOG_ROOT}/anchor_calibration.json"
readonly DEVICE="cuda:0"

export CUDA_VISIBLE_DEVICES="0"
readonly CUDA_VISIBLE_DEVICES
export CUBLAS_WORKSPACE_CONFIG=":4096:8"
export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
