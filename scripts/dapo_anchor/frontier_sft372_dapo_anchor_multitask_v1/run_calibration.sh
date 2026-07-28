#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

mkdir -p "${LOG_ROOT}"
if [[ -e "${CALIBRATION_REPORT}" ]]; then
  echo "Refusing to overwrite calibration: ${CALIBRATION_REPORT}" >&2
  exit 1
fi
if [[ -e "${LOG_ROOT}/calibration.stdout.log" ]]; then
  echo "Refusing to overwrite calibration log: ${LOG_ROOT}/calibration.stdout.log" >&2
  exit 1
fi

exec 9>"${LOG_ROOT}/calibration.lock"
if ! flock -n 9; then
  echo "Refusing concurrent calibration." >&2
  exit 1
fi
if [[ -e "${LOG_ROOT}/full_train.lock" ]]; then
  exec 8<>"${LOG_ROOT}/full_train.lock"
  if ! flock -n 8; then
    echo "Refusing calibration while formal training is active." >&2
    exit 1
  fi
fi

"${PYTHON}" "${SCRIPT_DIR}/calibrate.py" \
  --config "${CONFIG}" \
  --output "${CALIBRATION_REPORT}" \
  --device "${DEVICE}" \
  2>&1 | tee "${LOG_ROOT}/calibration.stdout.log"

