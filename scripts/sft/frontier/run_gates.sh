#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_python
"${SCRIPT_DIR}/run_train_stage.sh" frontier_gate_00512 512 1
"${SCRIPT_DIR}/run_train_stage.sh" frontier_gate_02048 2048 1
"${SCRIPT_DIR}/run_train_stage.sh" frontier_gate_08192 8192 1
"${SCRIPT_DIR}/run_train_stage.sh" frontier_gate_16384 16384 1

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PROJECT_ROOT}/src:${LLAMAFACTORY_ROOT}/src" \
    "${PYTHON}" -m ksllm4rec_sft.cli check-gates \
    --profile "${PROFILE}" \
    --artifact-lock "${ARTIFACT_LOCK}" \
    --log-root "${LOG_ROOT}" \
    --report "${GATES_REPORT}" \
    --project-root "${PROJECT_ROOT}" \
    --max-reserved-gib 20.0
