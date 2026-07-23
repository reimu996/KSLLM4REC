#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${PROJECT_ROOT}"

"${PYTHON}" -m ksllm4rec_rloo.temperature compare \
  --config "${CONFIG}" --expected-groups 512 --output-root "${OUTPUT_ROOT}" >/dev/null
"${PYTHON}" -m ksllm4rec_rloo.temperature verify \
  --config "${CONFIG}" --expected-groups 512 --output-root "${OUTPUT_ROOT}" \
  --smoke-root "${SMOKE_ROOT}" --require-determinism >/dev/null
