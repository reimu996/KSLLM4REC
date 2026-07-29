#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

"${PYTHON}" - "${PILOT_DIR}/pilot_report.json" "${CONFIG}" <<'PY'
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import yaml

from ksllm4rec_dapo_anchor_multitask_v1_1.config import validate_config

report = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
config = yaml.safe_load(Path(sys.argv[2]).read_text(encoding="utf-8"))
validate_config(config)
required = {
    "source_blocks",
    "source_groups",
    "rl_groups",
    "anchor_groups",
    "policy_optimizer_steps",
    "anchor_optimizer_steps",
    "parameters_changed",
    "peak_reserved_gib",
    "finite",
}
if set(report) != required:
    raise RuntimeError(f"Pilot report schema mismatch: {sorted(set(report) ^ required)}")
if int(report["source_blocks"]) < 1 or int(report["source_groups"]) < 1:
    raise RuntimeError("Pilot did not sample a source block.")
rl_groups = int(report["rl_groups"])
expected_steps = (rl_groups + int(config["loss"]["minibatch_groups"]) - 1) // int(
    config["loss"]["minibatch_groups"]
)
if rl_groups < int(config["sampling"]["minimum_effective_groups_per_update"]):
    raise RuntimeError("Pilot did not fill one optimization window.")
if int(report["policy_optimizer_steps"]) != expected_steps:
    raise RuntimeError("Pilot policy-step count differs from retained RL groups.")
if int(report["anchor_optimizer_steps"]) != 0:
    raise RuntimeError("Anchor must not create an independent optimizer step.")
if report["parameters_changed"] is not True or report["finite"] is not True:
    raise RuntimeError("Pilot did not produce a finite policy update.")
peak = float(report["peak_reserved_gib"])
if not math.isfinite(peak) or peak > float(config["memory"]["max_reserved_gib"]):
    raise RuntimeError("Pilot failed the reserved-memory gate.")
print(json.dumps({"passed": True, "pilot_report": sys.argv[1]}, separators=(",", ":")))
PY

