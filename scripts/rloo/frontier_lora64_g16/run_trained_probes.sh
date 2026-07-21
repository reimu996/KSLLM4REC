#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}/probes"

for epoch in 001 002; do
  "${PYTHON}" -m ksllm4rec_rloo.cli --config "${CONFIG}" probe \
    --groups "${GROUPS_FILE}" --calibration-ids "${CALIBRATION_IDS}" \
    --policy-adapter "${RUN_DIR}/epoch_${epoch}" --trie-dir "${TRIE_DIR}" \
    --output-dir "${LOG_ROOT}/probes/epoch_${epoch}" --device "${DEVICE}" \
    2> >(tee "${LOG_ROOT}/probe_epoch_${epoch}.stderr.log" >&2) \
    | tee "${LOG_ROOT}/probe_epoch_${epoch}.json"
done
