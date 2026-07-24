#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
readonly ROOT

# Only the Python executable is intentionally overridable. Every value that
# identifies this training profile is frozen here so an inherited environment
# cannot redirect gates or training to another config, device, or artifact.
readonly PYTHON="${PYTHON:-/home/lyc/miniconda3/envs/onereason_lora_sft/bin/python}"
readonly CONFIG="${ROOT}/configs/rloo/frontier_sft_lora64_lr1p5em4_wd1em3_step372_g16_rloo_dapo_t12_k1_kvcache.yaml"
readonly LOG_ROOT="${ROOT}/operation_logs/rloo/frontier_sft_lora64_lr1p5em4_wd1em3_step372_rloo_dapo_t12_k1_kvcache_v3"
readonly GATE_ROOT="${LOG_ROOT}/gates"
readonly PILOT_DIR="${ROOT}/artifacts/rloo/pilots/frontier_sft_lora64_lr1p5em4_wd1em3_step372_g16_t12_k1_kvcache_v3"
readonly RUN_DIR="${ROOT}/artifacts/rloo/runs/frontier_sft_lora64_lr1p5em4_wd1em3_step372_rloo_dapo_t12_k1_kvcache_2effective_epochs"
readonly TRAIN_LOG="${LOG_ROOT}/full_train.stdout.log"
readonly CPU_REPORT="${LOG_ROOT}/cpu_gate.json"
readonly CPU_LOG="${LOG_ROOT}/cpu_tests.log"
readonly DEVICE="cuda:0"
export CUDA_VISIBLE_DEVICES="0"
readonly CUDA_VISIBLE_DEVICES

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
