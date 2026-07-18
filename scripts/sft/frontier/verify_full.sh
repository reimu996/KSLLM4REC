#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_python
RUN_ID="frontier_verify_full_$(date '+%Y%m%d_%H%M%S_%N')"
RUN_DIR="${LOG_ROOT}/${RUN_ID}"
mkdir -p "${RUN_DIR}"
exec > >(tee -a "${RUN_DIR}/console.log") 2>&1
export_offline_runtime

"${PYTHON}" -m ksllm4rec_sft.cli verify \
    --profile "${PROFILE}" \
    --model "${MODEL}" \
    --output-dir "${FULL_OUTPUT}" \
    --log-root "${LOG_ROOT}" \
    --report "${RUN_DIR}/verification.json" \
    --artifact-lock "${ARTIFACT_LOCK}" \
    --environment-lock "${ENVIRONMENT_LOCK}" \
    --min-free-gib 20.5 \
    --max-reserved-gib 20.0
