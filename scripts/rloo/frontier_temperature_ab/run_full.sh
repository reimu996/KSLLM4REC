#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
source "${SCRIPT_DIR}/common.sh"
cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}"

run_arm_if_missing() {
  local arm="$1"
  local temperature="$2"
  local audit="${OUTPUT_ROOT}/${arm}_audit.jsonl"
  local summary="${OUTPUT_ROOT}/${arm}_summary.json"
  if [[ -f "${audit}" && -f "${summary}" ]]; then
    echo "Reusing completed arm: ${arm}" >&2
    return
  fi
  if [[ -e "${audit}" || -e "${summary}" ]]; then
    echo "Incomplete arm output requires inspection: ${arm}" >&2
    exit 1
  fi
  "${PYTHON}" -m ksllm4rec_rloo.temperature run \
    --config "${CONFIG}" --temperature "${temperature}" --max-groups 512 \
    --output-root "${OUTPUT_ROOT}" --device "${DEVICE}" >/dev/null
}

run_arm_if_missing t100 1.0
run_arm_if_missing t120 1.2
