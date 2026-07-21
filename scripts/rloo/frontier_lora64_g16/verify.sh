#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${PROJECT_ROOT}"

"${PYTHON}" -m ksllm4rec_rloo.cli --config "${CONFIG}" verify \
  --groups "${GROUPS_FILE}" --trie-dir "${TRIE_DIR}" \
  --calibration-ids "${CALIBRATION_IDS}" \
  --run-dir "${RUN_DIR}" --probe-root "${LOG_ROOT}/probes" \
  --gate-root "${LOG_ROOT}" \
  2> >(tee "${LOG_ROOT}/final_verification.stderr.log" >&2) \
  | tee "${LOG_ROOT}/final_verification.json"
