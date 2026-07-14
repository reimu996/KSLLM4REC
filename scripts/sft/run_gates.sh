#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ENV_NAME="onereason_lora_sft"
PYTHON="/home/lyc/miniconda3/envs/${ENV_NAME}/bin/python"

"${SCRIPT_DIR}/run_preflight.sh"
"${SCRIPT_DIR}/run_config_check.sh"
"${SCRIPT_DIR}/run_train_stage.sh" gate_00512 512 1
"${SCRIPT_DIR}/run_train_stage.sh" gate_02048 2048 1
"${SCRIPT_DIR}/run_train_stage.sh" gate_08192 8192 1
"${SCRIPT_DIR}/run_train_stage.sh" gate_16384 16384 1

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PROJECT_ROOT}/src" \
    "${PYTHON}" -m ksllm4rec_sft.cli check-gates \
    --log-root "${PROJECT_ROOT}/operation_logs/sft" \
    --report "${PROJECT_ROOT}/artifacts/sft/gates_report.json" \
    --project-root "${PROJECT_ROOT}" \
    --max-reserved-gib 20.0
