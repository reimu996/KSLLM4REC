#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

"${SCRIPT_DIR}/run_config_check.sh"
"${SCRIPT_DIR}/run_train_stage.sh" gate_00512
"${SCRIPT_DIR}/run_train_stage.sh" gate_02048
"${SCRIPT_DIR}/run_train_stage.sh" gate_08192
"${SCRIPT_DIR}/run_train_stage.sh" gate_16384
"${PYTHON}" -m ksllm4rec_orpo.cli check-gates \
    --log-root "${LOG_ROOT}" \
    --report "${GATES_REPORT}" \
    --project-root "${PROJECT_ROOT}" \
    --max-reserved-gib 20.0
