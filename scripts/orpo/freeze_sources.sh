#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
RUN_DIR="${LOG_ROOT}/freeze_sources_$(date '+%Y%m%d_%H%M%S_%N')"
mkdir -p "${RUN_DIR}" "${DATA_DIR}" "$(dirname -- "${GENERATION_LOCK}")"
exec > >(tee -a "${RUN_DIR}/console.log") 2>&1
trap 'status=$?; printf "exit_code=%s\nfinished_at=%s\n" "${status}" "$(date --iso-8601=seconds)" > "${RUN_DIR}/shell-status.txt"' EXIT

if [[ -f "${GENERATION_LOCK}" ]]; then
    "${PYTHON}" -m ksllm4rec_orpo.cli verify-sources --lock "${GENERATION_LOCK}"
else
    "${PYTHON}" -m ksllm4rec_orpo.cli freeze-sources \
        --lock "${GENERATION_LOCK}" \
        --model "${BASE_MODEL}" \
        --source "${SOURCE_DATA}" \
        --pid2sid "${PID2SID_DIR}" \
        --caption "${CAPTION_DIR}" \
        --text-probe "${TEXT_PROBE_DIR}" \
        --recommend-probe "${RECOMMEND_PROBE_DIR}" \
        --llamafactory "${LLAMAFACTORY_ROOT}"
fi
