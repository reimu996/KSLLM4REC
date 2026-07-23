#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${PROJECT_ROOT}"

PASS1="${SMOKE_ROOT}/pass1_t100_then_t120"
PASS2="${SMOKE_ROOT}/pass2_t120_then_t100"
for path in "${PASS1}" "${PASS2}"; do
  if [[ -e "${path}" ]]; then
    echo "Refusing to overwrite smoke output: ${path}" >&2
    exit 1
  fi
done
for path in "${PASS1}" "${PASS2}"; do
  mkdir -p "${path}"
done

"${PYTHON}" -m ksllm4rec_rloo.temperature run \
  --config "${CONFIG}" --temperature 1.0 --max-groups 8 \
  --output-root "${PASS1}" --device "${DEVICE}" >/dev/null
"${PYTHON}" -m ksllm4rec_rloo.temperature run \
  --config "${CONFIG}" --temperature 1.2 --max-groups 8 \
  --output-root "${PASS1}" --device "${DEVICE}" >/dev/null
"${PYTHON}" -m ksllm4rec_rloo.temperature compare \
  --config "${CONFIG}" --expected-groups 8 --output-root "${PASS1}" >/dev/null
"${PYTHON}" -m ksllm4rec_rloo.temperature verify \
  --config "${CONFIG}" --expected-groups 8 --output-root "${PASS1}" >/dev/null

"${PYTHON}" -m ksllm4rec_rloo.temperature run \
  --config "${CONFIG}" --temperature 1.2 --max-groups 8 \
  --output-root "${PASS2}" --device "${DEVICE}" >/dev/null
"${PYTHON}" -m ksllm4rec_rloo.temperature run \
  --config "${CONFIG}" --temperature 1.0 --max-groups 8 \
  --output-root "${PASS2}" --device "${DEVICE}" >/dev/null
"${PYTHON}" -m ksllm4rec_rloo.temperature compare \
  --config "${CONFIG}" --expected-groups 8 --output-root "${PASS2}" >/dev/null
"${PYTHON}" -m ksllm4rec_rloo.temperature verify \
  --config "${CONFIG}" --expected-groups 8 --output-root "${PASS2}" >/dev/null

"${PYTHON}" -m ksllm4rec_rloo.temperature determinism \
  --left-root "${PASS1}" --right-root "${PASS2}" --expected-groups 8 \
  --output "${SMOKE_ROOT}/determinism.json" >/dev/null
