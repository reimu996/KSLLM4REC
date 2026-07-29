#!/usr/bin/env bash
set -euo pipefail

source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

readonly PROBE_REPORT="${KSLLM4REC_PROBE_OUTPUT:-${ROOT}/artifacts/rloo/evaluations/dapo_anchor_multitask_sft372_v1_1_e2${ARM_SUFFIX}/probe64_report.json}"

exec "${PYTHON}" "${SCRIPT_DIR}/verify_probe64.py" \
  --config "${CONFIG}" \
  --report "${PROBE_REPORT}" \
  --require-complete
