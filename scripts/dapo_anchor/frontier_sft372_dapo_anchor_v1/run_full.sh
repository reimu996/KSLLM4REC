#!/usr/bin/env bash
# Launch formal training for the dapo_anchor SFT372 profile.
# Writes stdout/stderr to ${LOG_ROOT}/full_train.stdout.log via tee and holds
# an exclusive flock for the duration of the process so a second invocation
# refuses to start.
set -euo pipefail

source "$(dirname "$0")/common.sh"

mkdir -p "${LOG_ROOT}"

exec 9>"${LOG_ROOT}/full_train.lock"
if ! flock -n 9; then
  echo "Refusing concurrent training: ${LOG_ROOT}/full_train.lock is held." \
    | tee -a "${TRAIN_LOG}" >&2
  exit 1
fi

START_MARKER='{"event":"train_invocation_start","unix_time":'"$(date +%s)"',"source":"run_full.sh"}'
printf '%s\n' "${START_MARKER}" | tee -a "${TRAIN_LOG}"

set +e
"${PYTHON}" "${LAUNCHER}" \
  --config "${CONFIG}" \
  --output-dir "${RUN_DIR}" \
  --device "${DEVICE}" \
  2>&1 | tee -a "${TRAIN_LOG}"
train_status=${PIPESTATUS[0]}
set -e

END_MARKER='{"event":"train_invocation_end","unix_time":'"$(date +%s)"',"exit_code":'"${train_status}"',"source":"run_full.sh"}'
printf '%s\n' "${END_MARKER}" | tee -a "${TRAIN_LOG}"

exit "${train_status}"
