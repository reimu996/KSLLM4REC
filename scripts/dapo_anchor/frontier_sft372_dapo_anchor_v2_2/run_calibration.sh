#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
mkdir -p "${LOG_ROOT}"

exec 9>"${LOG_ROOT}/calibration.lock"
if ! flock -n 9; then
  echo "Refusing concurrent calibration: ${LOG_ROOT}/calibration.lock is held." >&2
  exit 1
fi

exec 8>"${LOG_ROOT}/full_train.lock"
if ! flock -n 8; then
  echo "Refusing calibration while formal training is active." >&2
  exit 1
fi

"${PYTHON}" "${CALIBRATION_LAUNCHER}" \
  --config "${CONFIG}" \
  --output "${CALIBRATION_REPORT}" \
  --device "${DEVICE}" \
  2>&1 | tee -a "${LOG_ROOT}/calibration.stdout.log"
