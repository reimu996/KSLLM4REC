#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

mkdir -p "${LOG_ROOT}"

exec 9>"${LOG_ROOT}/run_gates.lock"
if ! flock -n 9; then
  echo "Refusing concurrent gates: ${LOG_ROOT}/run_gates.lock is held." >&2
  exit 1
fi

run_cli cpu-check \
  --config "${CONFIG}" \
  --report "${CPU_REPORT}" \
  --log "${CPU_LOG}"

# Check every formal publication target before creating a staging path.
require_new_directory "${GATE_ROOT}"
require_new_directory "${PILOT_DIR}"
require_new_directory "${PILOT_DIR}-dense-baseline"
require_new_directory "${PILOT_DIR}-resume-check"

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ).$$"
STAGING_ROOT="${LOG_ROOT}/.gates.staging.${RUN_ID}"
FAILED_ROOT="${LOG_ROOT}/gates.failed.${RUN_ID}"
REPORT_STAGING="${STAGING_ROOT}/reports"
PILOT_STAGING="${STAGING_ROOT}/pilot"
readonly RUN_ID STAGING_ROOT FAILED_ROOT REPORT_STAGING PILOT_STAGING

require_new_directory "${STAGING_ROOT}"
require_new_directory "${FAILED_ROOT}"
mkdir -p "${REPORT_STAGING}"

rewrite_pilot_report_paths() {
  local report_path="$1"
  local old_prefix="$2"
  local new_prefix="$3"
  "${PYTHON}" - "${report_path}" "${old_prefix}" "${new_prefix}" <<'PY'
import json
import os
from pathlib import Path
import sys
import tempfile

report_path = Path(sys.argv[1])
old_prefix = sys.argv[2]
new_prefix = sys.argv[3]

def rewrite(value):
    if isinstance(value, str):
        return value.replace(old_prefix, new_prefix)
    if isinstance(value, list):
        return [rewrite(item) for item in value]
    if isinstance(value, dict):
        return {key: rewrite(item) for key, item in value.items()}
    return value

report = rewrite(json.loads(report_path.read_text(encoding="utf-8")))
descriptor, temporary = tempfile.mkstemp(
    prefix=f".{report_path.name}.", dir=report_path.parent
)
try:
    with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, report_path)
finally:
    if os.path.exists(temporary):
        os.unlink(temporary)
PY
}

published=false
preserve_failed_staging() {
  local status=$?
  trap - EXIT
  if [[ "${published}" != true ]]; then
    # Roll back either pilot directory if publication was interrupted midway.
    if [[ -e "${PILOT_DIR}" && ! -e "${PILOT_STAGING}" ]]; then
      mv -T "${PILOT_DIR}" "${PILOT_STAGING}" || true
    fi
    if [[ -e "${PILOT_DIR}-dense-baseline" && ! -e "${PILOT_STAGING}-dense-baseline" ]]; then
      mv -T "${PILOT_DIR}-dense-baseline" "${PILOT_STAGING}-dense-baseline" || true
    fi
    if [[ -e "${PILOT_DIR}-resume-check" && ! -e "${PILOT_STAGING}-resume-check" ]]; then
      mv -T "${PILOT_DIR}-resume-check" "${PILOT_STAGING}-resume-check" || true
    fi
    if [[ -e "${GATE_ROOT}" && ! -e "${REPORT_STAGING}" ]]; then
      mv -T "${GATE_ROOT}" "${REPORT_STAGING}" || true
    fi
    if [[ -f "${REPORT_STAGING}/pilot.json" ]]; then
      rewrite_pilot_report_paths \
        "${REPORT_STAGING}/pilot.json" "${PILOT_DIR}" "${PILOT_STAGING}" || true
    fi
    if [[ -e "${STAGING_ROOT}" ]]; then
      mv -T "${STAGING_ROOT}" "${FAILED_ROOT}" || true
      echo "Gate run failed; audit staging retained at: ${FAILED_ROOT}" >&2
    fi
  fi
  exit "${status}"
}
trap preserve_failed_staging EXIT
trap 'exit 129' HUP
trap 'exit 130' INT
trap 'exit 143' TERM

run_gate_sequence() {
  run_cli signature \
    --config "${CONFIG}" \
    --output "${REPORT_STAGING}/signature.json" || return $?
  run_cli structure-gate \
    --config "${CONFIG}" \
    --output "${REPORT_STAGING}/structure.json" || return $?
  run_cli probability-gate \
    --config "${CONFIG}" \
    --output "${REPORT_STAGING}/probability.json" \
    --device "${DEVICE}" || return $?
  run_cli memory-gate \
    --config "${CONFIG}" \
    --output "${REPORT_STAGING}/memory.json" \
    --device "${DEVICE}" || return $?
  run_cli throughput-gate \
    --config "${CONFIG}" \
    --output "${REPORT_STAGING}/throughput.json" \
    --device "${DEVICE}" || return $?
  run_cli pilot-gate \
    --config "${CONFIG}" \
    --output "${REPORT_STAGING}/pilot.json" \
    --pilot-dir "${PILOT_STAGING}" \
    --device "${DEVICE}" || return $?
}

run_gate_sequence 2>&1 | tee "${REPORT_STAGING}/run_gates.log"

# Recompute the current signature and validate all five staged reports together.
"${PYTHON}" - "${CONFIG}" "${REPORT_STAGING}" <<'PY'
import json
from pathlib import Path
import sys

from ksllm4rec_rloo_dapo.config import load_config
from ksllm4rec_rloo_dapo.device import gpu_identity
from ksllm4rec_rloo_dapo.fingerprint import runtime_signature
from ksllm4rec_rloo_dapo.gates import load_and_validate_gate_reports

config_path = Path(sys.argv[1])
report_root = Path(sys.argv[2])
config = load_config(config_path)
signature = runtime_signature(config, config_path=config_path)
written_signature = json.loads(
    (report_root / "signature.json").read_text(encoding="utf-8")
)
if written_signature != signature:
    raise RuntimeError("Staged signature no longer matches the current runtime.")
load_and_validate_gate_reports(
    config,
    signature,
    {
        name: report_root / f"{name}.json"
        for name in ("structure", "probability", "memory", "throughput", "pilot")
    },
    expected_gpu_identity=gpu_identity("cuda:0"),
)
PY

# Publish pilot artifacts first. The formal gate directory remains absent until
# all five reports have passed and the report paths refer to their final homes.
mkdir -p "$(dirname "${PILOT_DIR}")"
mv -T "${PILOT_STAGING}-dense-baseline" "${PILOT_DIR}-dense-baseline"
mv -T "${PILOT_STAGING}-resume-check" "${PILOT_DIR}-resume-check"
mv -T "${PILOT_STAGING}" "${PILOT_DIR}"

rewrite_pilot_report_paths \
  "${REPORT_STAGING}/pilot.json" "${PILOT_STAGING}" "${PILOT_DIR}"

# REPORT_STAGING and GATE_ROOT share a filesystem under LOG_ROOT, so this
# directory rename is the single atomic publication point for formal gates.
mv -T "${REPORT_STAGING}" "${GATE_ROOT}"
published=true
trap - EXIT HUP INT TERM
rmdir "${STAGING_ROOT}"
echo "Published validated gates atomically at: ${GATE_ROOT}"
