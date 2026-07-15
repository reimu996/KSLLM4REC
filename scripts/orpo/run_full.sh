#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

"${PYTHON}" -m ksllm4rec_orpo.cli check-gates \
    --log-root "${LOG_ROOT}" \
    --report "${GATES_REPORT}" \
    --project-root "${PROJECT_ROOT}" \
    --max-reserved-gib 20.0
"${SCRIPT_DIR}/run_train_stage.sh" full_orpo_epoch_002 "${FULL_OUTPUT}"
"${SCRIPT_DIR}/verify_full.sh"
