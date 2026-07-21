#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${PROJECT_ROOT}"
mkdir -p "${LOG_ROOT}/gate_runs"

"${PYTHON}" -m ksllm4rec_rloo.cli --config "${CONFIG}" structure-gate \
  --groups "${GROUPS_FILE}" --trie-dir "${TRIE_DIR}" \
  --calibration-ids "${CALIBRATION_IDS}" \
  --output "${LOG_ROOT}/structure_gate.json" \
  | tee "${LOG_ROOT}/structure_gate.stdout.json"

"${PYTHON}" -m ksllm4rec_rloo.cli --config "${CONFIG}" calibration-gate \
  --groups "${GROUPS_FILE}" --trie-dir "${TRIE_DIR}" \
  --calibration-ids "${CALIBRATION_IDS}" \
  --output "${LOG_ROOT}/calibration_gate.json" --device "${DEVICE}" \
  2> >(tee "${LOG_ROOT}/calibration_gate.stderr.log" >&2) \
  | tee "${LOG_ROOT}/calibration_gate.stdout.json"

"${PYTHON}" -m ksllm4rec_rloo.cli --config "${CONFIG}" probability-gate \
  --groups "${GROUPS_FILE}" --trie-dir "${TRIE_DIR}" \
  --calibration-ids "${CALIBRATION_IDS}" \
  --calibration-report "${LOG_ROOT}/calibration_gate.json" \
  --output "${LOG_ROOT}/probability_gate.json" --device "${DEVICE}" \
  2> >(tee "${LOG_ROOT}/probability_gate.stderr.log" >&2) \
  | tee "${LOG_ROOT}/probability_gate.stdout.json"

"${PYTHON}" -m ksllm4rec_rloo.cli --config "${CONFIG}" memory-gate \
  --groups "${GROUPS_FILE}" --trie-dir "${TRIE_DIR}" \
  --calibration-ids "${CALIBRATION_IDS}" \
  --calibration-report "${LOG_ROOT}/calibration_gate.json" \
  --output "${LOG_ROOT}/memory_gate.json" --device "${DEVICE}" \
  2> >(tee "${LOG_ROOT}/memory_gate.stderr.log" >&2) \
  | tee "${LOG_ROOT}/memory_gate.stdout.json"

"${PYTHON}" -m ksllm4rec_rloo.cli --config "${CONFIG}" signal-gate \
  --groups "${GROUPS_FILE}" --trie-dir "${TRIE_DIR}" \
  --calibration-ids "${CALIBRATION_IDS}" \
  --calibration-report "${LOG_ROOT}/calibration_gate.json" \
  --output-dir "${LOG_ROOT}/gate_runs/signal_512" \
  --output "${LOG_ROOT}/signal_gate.json" --device "${DEVICE}" \
  2> >(tee "${LOG_ROOT}/signal_gate.stderr.log" >&2) \
  | tee "${LOG_ROOT}/signal_gate.stdout.json"

"${PYTHON}" -m ksllm4rec_rloo.cli --config "${CONFIG}" timing-gate \
  --groups "${GROUPS_FILE}" --trie-dir "${TRIE_DIR}" \
  --calibration-ids "${CALIBRATION_IDS}" \
  --calibration-report "${LOG_ROOT}/calibration_gate.json" \
  --output-dir "${LOG_ROOT}/gate_runs/timing_256" \
  --output "${LOG_ROOT}/timing_gate.json" --device "${DEVICE}" \
  2> >(tee "${LOG_ROOT}/timing_gate.stderr.log" >&2) \
  | tee "${LOG_ROOT}/timing_gate.stdout.json"
