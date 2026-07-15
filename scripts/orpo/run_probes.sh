#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
PROBE="${DATA_DIR}/fixed_probe_1024.jsonl"
PROBE_ROOT="${PROJECT_ROOT}/artifacts/orpo/probes"
mapfile -t CHECKPOINTS < <(find "${FULL_OUTPUT}" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V)
if [[ ${#CHECKPOINTS[@]} -ne 2 ]]; then
    echo "Expected exactly two epoch checkpoints, found ${#CHECKPOINTS[@]}." >&2
    exit 2
fi

"${PYTHON}" -m ksllm4rec_orpo.cli evaluate-probe \
    --model "${BASE_MODEL}" --probe "${PROBE}" \
    --output-dir "${PROBE_ROOT}/base" --batch-size 4 --min-free-gib 20.5
"${PYTHON}" -m ksllm4rec_orpo.cli evaluate-probe \
    --model "${BASE_MODEL}" --adapter "${CHECKPOINTS[0]}" --probe "${PROBE}" \
    --output-dir "${PROBE_ROOT}/epoch_001" --batch-size 4 --min-free-gib 20.5
"${PYTHON}" -m ksllm4rec_orpo.cli evaluate-probe \
    --model "${BASE_MODEL}" --adapter "${CHECKPOINTS[1]}" --probe "${PROBE}" \
    --output-dir "${PROBE_ROOT}/epoch_002" --batch-size 4 --min-free-gib 20.5
