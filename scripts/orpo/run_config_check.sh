#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
mkdir -p "${PROJECT_ROOT}/artifacts/orpo"
"${PYTHON}" -m ksllm4rec_orpo.cli config-check \
    --config "${CONFIG}" \
    --artifact-lock "${ARTIFACT_LOCK}" \
    --environment-lock "${ENVIRONMENT_LOCK}" \
    --report "${PROJECT_ROOT}/artifacts/orpo/config_check.json"
