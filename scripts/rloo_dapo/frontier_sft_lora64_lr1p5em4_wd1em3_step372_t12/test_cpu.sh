#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

cd "${ROOT}"
mkdir -p "${LOG_ROOT}"

require_new_directory "${CPU_REPORT}"
require_new_directory "${CPU_LOG}"
run_cli cpu-gate \
  --config "${CONFIG}" \
  --output "${CPU_REPORT}" \
  --log "${CPU_LOG}"
