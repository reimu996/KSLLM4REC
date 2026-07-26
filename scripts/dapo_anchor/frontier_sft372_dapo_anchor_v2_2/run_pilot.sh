#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"

"$(dirname "$0")/run_calibration.sh"
exec 9>"${LOG_ROOT}/full_train.lock"
if ! flock -n 9; then
  echo "Refusing pilot while formal training is active." >&2
  exit 1
fi
if [[ -e "${PILOT_DIR}" ]]; then
  echo "Pilot directory already exists; refusing to overwrite: ${PILOT_DIR}" >&2
  exit 1
fi

"${PYTHON}" "$(dirname "$0")/pilot.py" \
  --config "${CONFIG}" \
  --output-dir "${PILOT_DIR}" \
  --device "${DEVICE}"
