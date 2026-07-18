#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"

require_python
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="${PROJECT_ROOT}/src:${LLAMAFACTORY_ROOT}/src" \
    "${PYTHON}" -m ksllm4rec_sft.cli preflight \
    --profile "${PROFILE}" \
    --model "${MODEL}" \
    --data "${DATA}" \
    --report "${DATA_DIR}/tokenizer_report.json" \
    --artifact-lock "${ARTIFACT_LOCK}" \
    --cutoff-len 16384
