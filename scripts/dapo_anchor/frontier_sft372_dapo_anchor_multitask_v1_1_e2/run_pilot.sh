#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

if [[ -e "${PILOT_DIR}" ]]; then
  echo "Refusing to overwrite pilot directory: ${PILOT_DIR}" >&2
  exit 1
fi
if [[ -e "${LOG_ROOT}/pilot.stdout.log" ]]; then
  echo "Refusing to overwrite pilot log: ${LOG_ROOT}/pilot.stdout.log" >&2
  exit 1
fi
mkdir -p "${LOG_ROOT}"

exec 9>"${LOG_ROOT}/pilot.lock"
if ! flock -n 9; then
  echo "Refusing concurrent pilot." >&2
  exit 1
fi
if [[ -e "${LOG_ROOT}/full_train.lock" ]]; then
  exec 8<>"${LOG_ROOT}/full_train.lock"
  if ! flock -n 8; then
    echo "Refusing pilot while formal training is active." >&2
    exit 1
  fi
fi

"${PYTHON}" "${SCRIPT_DIR}/pilot.py" \
  --config "${CONFIG}" \
  --output-dir "${PILOT_DIR}" \
  --device "${DEVICE}" \
  2>&1 | tee "${LOG_ROOT}/pilot.stdout.log"
"${SCRIPT_DIR}/verify_pilot.sh"
