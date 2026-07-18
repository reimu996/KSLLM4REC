#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "${PROJECT_ROOT}"

mkdir -p "${LOG_ROOT}/probes"
for epoch in 001 002; do
  "${PYTHON}" -m ksllm4rec_grpo.cli --config "${CONFIG}" probe \
    --policy-adapter "${RUN_DIR}/epoch_${epoch}" --trie-dir "${TRIE_DIR}" \
    --output-dir "${LOG_ROOT}/probes/epoch_${epoch}" \
    2> >(tee "${LOG_ROOT}/probe_epoch_${epoch}.stderr.log" >&2) \
    | tee "${LOG_ROOT}/probe_epoch_${epoch}.json"
done
