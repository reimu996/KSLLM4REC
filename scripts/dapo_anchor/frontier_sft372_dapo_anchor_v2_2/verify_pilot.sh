#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
"${PYTHON}" - "${PILOT_DIR}/pilot_report.json" <<'PY'
import json
import sys
from pathlib import Path

from ksllm4rec_dapo_anchor_v2_2.verify import validate_window_log_row

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if report.get("passed") is not True or report.get("formal_recovery_point") is not False:
    raise RuntimeError("Pilot report status is invalid.")
window = report["window"]
if report.get("logical_window_index") != 0 or window.get("window_index") != 0:
    raise RuntimeError("V2.2 pilot must exercise logical window W0.")
validate_window_log_row(window)
if window["effective_groups"] != 32 or window["rl_optimizer_steps"] != 4:
    raise RuntimeError("Pilot did not complete the 32-group/four-step RL window.")
if window["anchor_candidate_groups"] != window["anchor_groups_used"]:
    raise RuntimeError("Pilot did not use every anchor candidate.")
if window["anchor_candidate_groups"] and window["anchor_optimizer_steps"] != 1:
    raise RuntimeError("Pilot anchor candidates did not produce exactly one step.")
if not report["parameter_changed"]:
    raise RuntimeError("Pilot parameters did not change.")
print(json.dumps({"passed": True, "pilot_report": sys.argv[1]}, separators=(",", ":")))
PY
