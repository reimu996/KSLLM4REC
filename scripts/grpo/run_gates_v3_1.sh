#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "${PROJECT_ROOT}"

mkdir -p "${LOG_ROOT}/gate_runs"
"${PYTHON}" -m ksllm4rec_grpo.cli --config "${CONFIG}" memory-gate \
  --groups "${DATA_DIR}/groups.jsonl" --trie-dir "${TRIE_DIR}" \
  2> >(tee "${LOG_ROOT}/memory_gate.stderr.log" >&2) \
  | tee "${LOG_ROOT}/memory_gate.json"
"${PYTHON}" -m ksllm4rec_grpo.cli --config "${CONFIG}" signal-gate \
  --groups "${DATA_DIR}/groups.jsonl" --trie-dir "${TRIE_DIR}" \
  --memory-report "${LOG_ROOT}/memory_gate.json" \
  --output-dir "${LOG_ROOT}/gate_runs/signal_512" \
  2> >(tee "${LOG_ROOT}/signal_gate.stderr.log" >&2) \
  | tee "${LOG_ROOT}/signal_gate.json"
"${PYTHON}" -m ksllm4rec_grpo.cli --config "${CONFIG}" timing-gate \
  --groups "${DATA_DIR}/groups.jsonl" --trie-dir "${TRIE_DIR}" \
  --memory-report "${LOG_ROOT}/memory_gate.json" \
  --output-dir "${LOG_ROOT}/gate_runs/timing_32" \
  2> >(tee "${LOG_ROOT}/timing_gate.stderr.log" >&2) \
  | tee "${LOG_ROOT}/timing_gate.json"
