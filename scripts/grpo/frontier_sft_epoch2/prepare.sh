#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}"

"${PYTHON}" -m ksllm4rec_grpo.cli --config "${CONFIG}" config-check \
  | tee "${LOG_ROOT}/config_check.json"

SOURCE_LOCK="${PROJECT_ROOT}/artifacts/grpo/data/frontier_source_lock_v1/source_manifest.json"
PROBE_FILE="${PROBE_DIR}/fixed_probe_1024.jsonl"
if [[ ! -f "${SOURCE_LOCK}" || ! -f "${DATA_DIR}/groups.jsonl" || \
      ! -f "${TRIE_DIR}/manifest.json" || ! -f "${PROBE_FILE}" ]]; then
  if [[ -d "${DATA_DIR}" || -d "${TRIE_DIR}" || -d "${PROBE_DIR}" ]]; then
    printf 'Frontier artifact set is partial; refusing implicit rebuild.\n' >&2
    exit 1
  fi
  "${PYTHON}" scripts/grpo/prepare_frontier_v2.py \
    | tee "${LOG_ROOT}/prepare_frontier_v2.json"
fi

"${PYTHON}" -m ksllm4rec_grpo.cli --config "${CONFIG}" prepare-data \
  --output-dir "${DATA_DIR}" | tee "${LOG_ROOT}/prepare_data.json"
"${PYTHON}" -m ksllm4rec_grpo.cli --config "${CONFIG}" build-trie \
  --output-dir "${TRIE_DIR}" | tee "${LOG_ROOT}/build_trie.json"

if [[ ! -f "${PROBE_FILE}" ]]; then
  printf 'Frontier fixed probe is missing: %s\n' "${PROBE_FILE}" >&2
  exit 1
fi
sha256sum "${SOURCE_LOCK}" "${DATA_DIR}/data_manifest.json" \
  "${TRIE_DIR}/manifest.json" "${PROBE_DIR}/probe_manifest.json" \
  | tee "${LOG_ROOT}/frontier_artifact_hashes.txt"
