#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

training_active=false
if [[ -e "${LOG_ROOT}/full_train.lock" ]]; then
  exec 8<>"${LOG_ROOT}/full_train.lock"
  if flock -n 8; then
    flock -u 8
  else
    training_active=true
  fi
fi

"${PYTHON}" - \
  "${training_active}" "${RUN_DIR}" "${TRAIN_LOG}" "${CONFIG}" "${CALIBRATION_REPORT}" <<'PY'
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import yaml


def last_jsonl(path: Path):
    if not path.is_file() or path.stat().st_size == 0:
        return None
    lines = path.read_bytes().splitlines()
    return json.loads(lines[-1]) if lines else None


active = sys.argv[1] == "true"
run_dir = Path(sys.argv[2])
train_log = Path(sys.argv[3])
config = yaml.safe_load(Path(sys.argv[4]).read_text(encoding="utf-8"))
calibration_path = Path(sys.argv[5])
total_windows = int(config["train"]["total_windows"])
expected_rl_steps = int(config["train"]["total_rl_optimizer_updates"])
max_total_steps = int(config["train"]["max_total_optimizer_updates"])

window_path = run_dir / "windows.jsonl"
window = last_jsonl(window_path)
summary_path = run_dir / "run_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
latest_path = run_dir / "recovery/latest.json"
latest = json.loads(latest_path.read_text(encoding="utf-8")) if latest_path.is_file() else None
calibration = (
    json.loads(calibration_path.read_text(encoding="utf-8"))
    if calibration_path.is_file()
    else None
)

tail = ""
if train_log.is_file():
    with train_log.open("rb") as handle:
        handle.seek(0, 2)
        end = handle.tell()
        handle.seek(max(0, end - 262_144))
        tail = handle.read().decode("utf-8", errors="replace")
errors = [
    marker
    for marker in ("Traceback (most recent call last)", "CUDA out of memory", "Killed")
    if marker in tail
]

completed = int(window["window_index"]) + 1 if window else 0
progress_age = time.time() - train_log.stat().st_mtime if train_log.is_file() else None
stalled = bool(active and progress_age is not None and progress_age > 900)
summary_valid = bool(
    isinstance(summary, dict)
    and summary.get("complete") is True
    and int(summary.get("completed_windows", -1)) == total_windows
    and int(summary.get("rl_optimizer_update_step", -1)) == expected_rl_steps
    and int(summary.get("optimizer_update_step", -1))
    == int(summary.get("rl_optimizer_update_step", -2))
    + int(summary.get("anchor_optimizer_update_step", -3))
    and int(summary.get("optimizer_update_step", max_total_steps + 1)) <= max_total_steps
)
success = bool(not active and not errors and not stalled and summary_valid)
if errors:
    state = "error"
elif stalled:
    state = "stalled"
elif active and completed >= total_windows:
    state = "finalizing"
elif active:
    state = "running"
elif success:
    state = "complete"
else:
    state = "incomplete"

report = {
    "state": state,
    "success": success,
    "training_active": active,
    "calibration_ready": isinstance(calibration, dict),
    "lambda_calibrated": calibration.get("lambda_calibrated") if calibration else None,
    "calibration_valid_windows": calibration.get("valid_windows") if calibration else None,
    "completed_windows": completed,
    "total_windows_expected": total_windows,
    "rl_optimizer_update_step": window.get("rl_optimizer_update_step") if window else 0,
    "anchor_optimizer_update_step": window.get("anchor_optimizer_update_step") if window else 0,
    "optimizer_update_step": window.get("optimizer_update_step") if window else 0,
    "last_anchor_candidate_groups": window.get("anchor_candidate_groups") if window else None,
    "last_anchor_groups_used": window.get("anchor_groups_used") if window else None,
    "last_anchor_raw_grad_norm": window.get("anchor_raw_grad_norm") if window else None,
    "last_rl_reference_grad_norm": window.get("rl_reference_grad_norm") if window else None,
    "last_lambda_effective": window.get("lambda_effective") if window else None,
    "last_anchor_to_rl_grad_ratio": window.get("anchor_to_rl_grad_ratio") if window else None,
    "last_window_seconds": window.get("window_seconds") if window else None,
    "last_peak_reserved_gib": window.get("peak_reserved_gib") if window else None,
    "last_progress_age_seconds": progress_age,
    "latest_recovery": latest,
    "run_summary": summary,
    "errors": errors,
}
print(json.dumps(report, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
if state in {"error", "stalled", "incomplete"}:
    raise SystemExit(1)
PY
