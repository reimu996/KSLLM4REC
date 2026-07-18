#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "${PROJECT_ROOT}"

mkdir -p "${LOG_ROOT}"
"${PYTHON}" -m ksllm4rec_grpo.cli --config "${CONFIG}" config-check \
  | tee "${LOG_ROOT}/config_check.json"
"${PYTHON}" -m ksllm4rec_grpo.cli --config "${CONFIG}" prepare-data \
  --output-dir "${DATA_DIR}" | tee "${LOG_ROOT}/prepare_data.json"
"${PYTHON}" -m ksllm4rec_grpo.cli --config "${CONFIG}" build-trie \
  --output-dir "${TRIE_DIR}" | tee "${LOG_ROOT}/build_trie.json"
