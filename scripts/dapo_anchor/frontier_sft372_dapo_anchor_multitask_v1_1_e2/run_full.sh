#!/usr/bin/env bash
# Formal V1.1 entry point. It deliberately starts no status/health polling loop.
set -euo pipefail
source "$(dirname "$0")/common.sh"

mkdir -p "${LOG_ROOT}"

if [[ -e "${RUN_DIR}" && ! -d "${RUN_DIR}" ]]; then
  echo "Run path exists and is not a directory: ${RUN_DIR}" >&2
  exit 1
fi
if [[ -d "${RUN_DIR}" ]] && find "${RUN_DIR}" -mindepth 1 -maxdepth 1 -print -quit | grep -q .; then
  for required in \
    resolved_config.json \
    runtime_signature.json \
    source_epoch_plan.json \
    recovery/latest.json; do
    if [[ ! -f "${RUN_DIR}/${required}" ]]; then
      echo "Refusing to overwrite non-recoverable run directory; missing ${required}." >&2
      exit 1
    fi
  done
fi

exec 9>"${LOG_ROOT}/full_train.lock"
if ! flock -n 9; then
  echo "Refusing concurrent training: ${LOG_ROOT}/full_train.lock is held." >&2
  exit 1
fi

exec "${PYTHON}" "${SCRIPT_DIR}/train.py" \
  --config "${CONFIG}" \
  --output-dir "${RUN_DIR}" \
  --device "${DEVICE}" \
  --formal
