#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "${PROJECT_ROOT}"

mkdir -p "${LOG_ROOT}"
"${PYTHON}" -m unittest discover -s tests/grpo -v \
  2>&1 | tee "${LOG_ROOT}/cpu_unittest.log"
"${PYTHON}" -m ksllm4rec_grpo.cli --config "${CONFIG}" test-tokenizer \
  --groups "${DATA_DIR}/groups.jsonl" --trie-dir "${TRIE_DIR}" \
  | tee "${LOG_ROOT}/tokenizer_gate.json"
