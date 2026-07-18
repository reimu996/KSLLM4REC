#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "${PROJECT_ROOT}"

"${PYTHON}" -m ksllm4rec_grpo.cli --config "${CONFIG}" train \
  --groups "${DATA_DIR}/groups.jsonl" --trie-dir "${TRIE_DIR}" \
  --memory-report "${LOG_ROOT}/memory_gate.json" \
  --signal-report "${LOG_ROOT}/signal_gate.json" \
  --timing-report "${LOG_ROOT}/timing_gate.json" \
  --output-dir "${RUN_DIR}" \
  2> >(tee "${LOG_ROOT}/full_train.stderr.log" >&2) \
  | tee "${LOG_ROOT}/full_train_result.json"
