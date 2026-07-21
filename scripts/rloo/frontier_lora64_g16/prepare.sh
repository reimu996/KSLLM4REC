#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}"

"${PYTHON}" -m ksllm4rec_rloo.cli --config "${CONFIG}" config-check \
  --groups "${GROUPS_FILE}" --trie-dir "${TRIE_DIR}" \
  | tee "${LOG_ROOT}/config_check.json"

"${PYTHON}" -m ksllm4rec_rloo.cli --config "${CONFIG}" prepare-calibration \
  --groups "${GROUPS_FILE}" --output "${CALIBRATION_IDS}" \
  | tee "${LOG_ROOT}/prepare_calibration.json"
