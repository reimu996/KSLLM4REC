#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_python
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PROJECT_ROOT}/src" \
    "${PYTHON}" -m ksllm4rec_sft.cli prepare-data \
    --profile "${PROFILE}" \
    --source "${SOURCE}" \
    --output-dir "${DATA_DIR}"

PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PROJECT_ROOT}/src:${LLAMAFACTORY_ROOT}/src" \
    "${PYTHON}" -m ksllm4rec_sft.cli create-lock \
    --profile "${PROFILE}" \
    --source "${SOURCE}" \
    --derived "${DATA}" \
    --base-lock "${BASE_ARTIFACT_LOCK}" \
    --output "${ARTIFACT_LOCK}"
