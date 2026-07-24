#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PYTHON="${PYTHON:-/home/lyc/miniconda3/envs/onereason_lora_sft/bin/python}"
CONFIG="${CONFIG:-${ROOT}/configs/rloo/frontier_sft_epoch2_lora64_g16_rloo_dapo_t12_k1_kvcache.yaml}"
LOG_ROOT="${LOG_ROOT:-${ROOT}/operation_logs/rloo/frontier_sft_epoch2_lora64_rloo_dapo_t12_k1_kvcache_v2}"
GATE_ROOT="${GATE_ROOT:-${LOG_ROOT}/gates}"
PILOT_DIR="${PILOT_DIR:-${ROOT}/artifacts/rloo/pilots/frontier_lora64_g16_t12_k1_kvcache_v2}"
RUN_DIR="${RUN_DIR:-${ROOT}/artifacts/rloo/runs/frontier_sft_epoch2_lora64_rloo_dapo_t12_k1_kvcache_2epoch}"
DEVICE="${DEVICE:-cuda:0}"

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

run_cli() {
  "${PYTHON}" -m ksllm4rec_rloo_dapo.cli "$@"
}

require_new_directory() {
  local path="$1"
  if [[ -e "${path}" ]]; then
    echo "Refusing to overwrite existing path: ${path}" >&2
    return 1
  fi
}
