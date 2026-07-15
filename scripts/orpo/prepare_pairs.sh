#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
RUN_DIR="${LOG_ROOT}/prepare_pairs_$(date '+%Y%m%d_%H%M%S_%N')"
mkdir -p "${RUN_DIR}" "${DATA_DIR}"
exec > >(tee -a "${RUN_DIR}/console.log") 2>&1
trap 'status=$?; printf "exit_code=%s\nfinished_at=%s\n" "${status}" "$(date --iso-8601=seconds)" > "${RUN_DIR}/shell-status.txt"' EXIT

if [[ ! -f "${DATA_DIR}/pair_manifest.json" ]]; then
    "${PYTHON}" -m ksllm4rec_orpo.cli prepare-candidates \
        --generation-lock "${GENERATION_LOCK}" \
        --source "${SOURCE_DATA}" \
        --pid2sid "${PID2SID_DIR}" \
        --caption "${CAPTION_DIR}" \
        --output-dir "${DATA_DIR}"
    "${PYTHON}" -m ksllm4rec_orpo.cli mine-pairs \
        --generation-lock "${GENERATION_LOCK}" \
        --candidates "${DATA_DIR}/candidates.jsonl" \
        --model "${BASE_MODEL}" \
        --output-dir "${DATA_DIR}" \
        --min-free-gib 20.5
fi

"${PYTHON}" -m ksllm4rec_orpo.cli preflight \
    --model "${BASE_MODEL}" \
    --data-dir "${DATA_DIR}" \
    --report "${DATA_DIR}/preflight_report.json" \
    --cutoff-len 16384

"${PYTHON}" -m ksllm4rec_orpo.cli prepare-probe \
    --baseline "${SOURCE_DATA}" \
    --text-to-sid "${TEXT_PROBE_DIR}" \
    --recommend "${RECOMMEND_PROBE_DIR}" \
    --output "${DATA_DIR}/fixed_probe_1024.jsonl" \
    --report "${DATA_DIR}/probe_report.json"

if [[ ! -f "${ARTIFACT_LOCK}" ]]; then
    "${PYTHON}" -m ksllm4rec_orpo.cli lock-training \
        --lock "${ARTIFACT_LOCK}" \
        --generation-lock "${GENERATION_LOCK}" \
        --model "${BASE_MODEL}" \
        --source "${SOURCE_DATA}" \
        --llamafactory "${LLAMAFACTORY_ROOT}" \
        --data-file "${DATA_DIR}/candidates.jsonl" \
        --data-file "${DATA_DIR}/candidate_report.json" \
        --data-file "${DATA_DIR}/audit.jsonl" \
        --data-file "${DATA_DIR}/train.jsonl" \
        --data-file "${DATA_DIR}/dataset_info.json" \
        --data-file "${DATA_DIR}/pair_manifest.json" \
        --data-file "${DATA_DIR}/memory_gate.jsonl" \
        --data-file "${DATA_DIR}/preflight_report.json" \
        --data-file "${DATA_DIR}/fixed_probe_1024.jsonl" \
        --data-file "${DATA_DIR}/probe_report.json"
fi
