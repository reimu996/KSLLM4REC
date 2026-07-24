#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

run_cli train \
  --config "${CONFIG}" \
  --structure-report "${GATE_ROOT}/structure.json" \
  --probability-report "${GATE_ROOT}/probability.json" \
  --memory-report "${GATE_ROOT}/memory.json" \
  --throughput-report "${GATE_ROOT}/throughput.json" \
  --pilot-report "${GATE_ROOT}/pilot.json" \
  --output-dir "${RUN_DIR}" \
  --device "${DEVICE}"
