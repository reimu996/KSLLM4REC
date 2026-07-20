#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${PROJECT_ROOT}"

"${PYTHON}" -m ksllm4rec_grpo.cli --config "${CONFIG}" verify \
  --groups "${DATA_DIR}/groups.jsonl" --trie-dir "${TRIE_DIR}" \
  --run-dir "${RUN_DIR}" --probe-root "${LOG_ROOT}/probes" \
  --gate-root "${LOG_ROOT}" \
  2> >(tee "${LOG_ROOT}/final_verification.stderr.log" >&2) \
  | tee "${LOG_ROOT}/final_verification.json"
