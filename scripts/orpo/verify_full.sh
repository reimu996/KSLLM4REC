#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
RUN_DIR="${LOG_ROOT}/verify_full_$(date '+%Y%m%d_%H%M%S_%N')"
mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/console.log") 2>&1
trap 'status=$?; printf "exit_code=%s\nfinished_at=%s\n" "${status}" "$(date --iso-8601=seconds)" > "${RUN_DIR}/shell-status.txt"' EXIT

"${PYTHON}" -m ksllm4rec_orpo.cli verify \
    --model "${BASE_MODEL}" \
    --output-dir "${FULL_OUTPUT}" \
    --log-root "${LOG_ROOT}" \
    --report "${RUN_DIR}/verification.json" \
    --artifact-lock "${ARTIFACT_LOCK}" \
    --environment-lock "${ENVIRONMENT_LOCK}" \
    --min-free-gib 20.5 \
    --max-reserved-gib 20.0
