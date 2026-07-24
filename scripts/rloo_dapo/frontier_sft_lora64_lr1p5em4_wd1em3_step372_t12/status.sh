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

"${PYTHON}" - "${training_active}" "${RUN_DIR}" "${TRAIN_LOG}" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys
import time


def last_line(path: Path) -> str | None:
    if not path.is_file() or path.stat().st_size == 0:
        return None
    with path.open("rb") as handle:
        handle.seek(0, 2)
        position = handle.tell()
        suffix = b""
        while position > 0:
            size = min(65_536, position)
            position -= size
            handle.seek(position)
            suffix = handle.read(size) + suffix
            lines = suffix.splitlines()
            if len(lines) >= 2 or position == 0:
                break
    if not suffix.endswith(b"\n"):
        lines = lines[:-1]
    return lines[-1].decode("utf-8", errors="replace") if lines else None


active = sys.argv[1] == "true"
run_dir = Path(sys.argv[2])
train_log = Path(sys.argv[3])
window_path = run_dir / "windows.jsonl"
window_raw = last_line(window_path)
window = json.loads(window_raw) if window_raw else None
log_raw = ""
if train_log.is_file():
    with train_log.open("rb") as handle:
        handle.seek(0, 2)
        end = handle.tell()
        handle.seek(max(0, end - 262_144))
        log_raw = handle.read().decode("utf-8", errors="replace")
invocation_marker = '{"event":"train_invocation_start"'
marker_offset = log_raw.rfind(invocation_marker)
invocation_log = log_raw[marker_offset:] if marker_offset >= 0 else log_raw
last_event = None
for line in reversed(invocation_log.splitlines()):
    if line.startswith("{"):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "event" in value:
            last_event = value
            break
errors = [
    marker
    for marker in ("Traceback (most recent call last)", "CUDA out of memory", "Killed")
    if marker in invocation_log
]
latest_path = run_dir / "recovery" / "latest.json"
latest = json.loads(latest_path.read_text(encoding="utf-8")) if latest_path.is_file() else None
summary_path = run_dir / "run_summary.json"
summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else None
completed = int(window["window_index"]) + 1 if window else 0
optimizer_update_step = window.get("optimizer_update_step") if window else 0
age = time.time() - window_path.stat().st_mtime if window_path.is_file() else None
progress_age = time.time() - train_log.stat().st_mtime if train_log.is_file() else None
stalled = bool(active and progress_age is not None and progress_age > 900)
invocation_exit_code = (
    last_event.get("exit_code")
    if isinstance(last_event, dict)
    and last_event.get("event") == "train_invocation_end"
    else None
)
summary_complete = bool(
    isinstance(summary, dict)
    and summary.get("complete") is True
    and summary.get("completed_windows") == 1064
    and summary.get("optimizer_update_step") == 4256
)
success = bool(
    not active
    and not errors
    and not stalled
    and completed == 1064
    and optimizer_update_step == 4256
    and summary_complete
    and invocation_exit_code == 0
)
if errors:
    state = "error"
elif stalled:
    state = "stalled"
elif active and completed >= 1064:
    state = "finalizing"
elif active:
    state = "running"
elif success:
    state = "complete"
elif invocation_exit_code not in (None, 0):
    state = "failed"
else:
    state = "incomplete"
report = {
    "state": state,
    "success": success,
    "training_active": active,
    "completed_windows": completed,
    "optimizer_update_step": optimizer_update_step,
    "last_window_seconds": window.get("window_seconds") if window else None,
    "last_raw_prompt_count": window.get("raw_prompt_count") if window else None,
    "last_peak_reserved_gib": window.get("peak_reserved_gib") if window else None,
    "last_window_age_seconds": age,
    "last_progress_age_seconds": progress_age,
    "stalled": stalled,
    "latest_recovery": latest,
    "run_summary": summary,
    "last_event": last_event,
    "invocation_exit_code": invocation_exit_code,
    "errors": errors,
}
print(json.dumps(report, ensure_ascii=False, separators=(",", ":"), allow_nan=False))
if state in {"error", "stalled", "failed", "incomplete"}:
    raise SystemExit(1)
PY
