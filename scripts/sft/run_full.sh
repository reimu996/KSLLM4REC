#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
ENV_NAME="onereason_lora_sft"
PYTHON="/home/lyc/miniconda3/envs/${ENV_NAME}/bin/python"
FULL_OUTPUT="${PROJECT_ROOT}/artifacts/sft/runs/full_epoch_001"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PROJECT_ROOT}/src" \
    "${PYTHON}" -m ksllm4rec_sft.cli check-gates \
    --log-root "${PROJECT_ROOT}/operation_logs/sft" \
    --report "${PROJECT_ROOT}/artifacts/sft/gates_report.json" \
    --project-root "${PROJECT_ROOT}" \
    --max-reserved-gib 21.5

"${SCRIPT_DIR}/run_train_stage.sh" full_epoch_001 32768 full "${FULL_OUTPUT}"
"${SCRIPT_DIR}/verify_full.sh"
