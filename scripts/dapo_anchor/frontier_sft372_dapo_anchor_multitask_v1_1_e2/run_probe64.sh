#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

readonly PROBE_ADAPTER="${KSLLM4REC_PROBE_ADAPTER:?Set KSLLM4REC_PROBE_ADAPTER to an adapter directory.}"
readonly PROBE_OUTPUT="${KSLLM4REC_PROBE_OUTPUT:-${ROOT}/artifacts/rloo/evaluations/dapo_anchor_multitask_sft372_v1_1_e2${ARM_SUFFIX}/probe64_report.json}"

if [[ -e "${PROBE_OUTPUT}" ]]; then
  echo "Refusing to overwrite Probe64 report: ${PROBE_OUTPUT}" >&2
  exit 1
fi

exec "${PYTHON}" "${SCRIPT_DIR}/run_probe64.py" \
  --config "${CONFIG}" \
  --adapter "${PROBE_ADAPTER}" \
  --output "${PROBE_OUTPUT}" \
  --device "${DEVICE}"
