#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

mkdir -p "${LOG_ROOT}"

exec 9>"${LOG_ROOT}/full_train.lock"
if ! flock -n 9; then
  echo "Refusing concurrent training: ${LOG_ROOT}/full_train.lock is held." \
    | tee -a "${TRAIN_LOG}" >&2
  exit 1
fi

printf '{"event":"train_invocation_start","unix_time":%s}\n' "$(date +%s)" \
  | tee -a "${TRAIN_LOG}"

set +e
run_cli train \
  --config "${CONFIG}" \
  --cpu-report "${CPU_REPORT}" \
  --structure-report "${GATE_ROOT}/structure.json" \
  --probability-report "${GATE_ROOT}/probability.json" \
  --memory-report "${GATE_ROOT}/memory.json" \
  --throughput-report "${GATE_ROOT}/throughput.json" \
  --pilot-report "${GATE_ROOT}/pilot.json" \
  --output-dir "${RUN_DIR}" \
  --device "${DEVICE}" \
  2>&1 | tee -a "${TRAIN_LOG}"
train_status=${PIPESTATUS[0]}
set -e
printf '{"event":"train_invocation_end","unix_time":%s,"exit_code":%s}\n' \
  "$(date +%s)" "${train_status}" | tee -a "${TRAIN_LOG}"
exit "${train_status}"
