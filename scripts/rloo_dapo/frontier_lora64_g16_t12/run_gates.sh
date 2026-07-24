#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

mkdir -p "${GATE_ROOT}"
run_cli structure-gate --config "${CONFIG}" --output "${GATE_ROOT}/structure.json"
run_cli probability-gate --config "${CONFIG}" --output "${GATE_ROOT}/probability.json" --device "${DEVICE}"
run_cli memory-gate --config "${CONFIG}" --output "${GATE_ROOT}/memory.json" --device "${DEVICE}"
run_cli throughput-gate --config "${CONFIG}" --output "${GATE_ROOT}/throughput.json" --device "${DEVICE}"

require_new_directory "${PILOT_DIR}"
run_cli pilot-gate \
  --config "${CONFIG}" \
  --output "${GATE_ROOT}/pilot.json" \
  --pilot-dir "${PILOT_DIR}" \
  --device "${DEVICE}"
