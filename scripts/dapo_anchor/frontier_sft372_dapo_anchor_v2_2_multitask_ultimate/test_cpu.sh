#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "${ROOT}"
PYTHONDONTWRITEBYTECODE=1 "${PYTHON}" -m compileall -q \
  src/ksllm4rec_dapo_anchor_v2_2_multitask_ultimate \
  scripts/dapo_anchor/frontier_sft372_dapo_anchor_v2_2_multitask
PYTHONDONTWRITEBYTECODE=1 "${PYTHON}" - "${CONFIG}" <<'PY'
import sys
from pathlib import Path

import yaml

sys.path.insert(0, "src")
from ksllm4rec_dapo_anchor_v2_2_multitask_ultimate import contract
from ksllm4rec_dapo_anchor_v2_2_multitask_ultimate.config import build_config
from ksllm4rec_dapo_anchor_v2_2_multitask_ultimate.trainer import load_groups_and_trie

actual = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
assert actual == build_config()
assert not {"calibration_windows", "min_calibration_windows", "calibration_path"} & set(actual["anchor"])
groups, _ = load_groups_and_trie(actual)
assert len(groups) == contract.TOTAL_MULTITASK_GROUPS
assert len({f"{group.task.value}|{group.group_id}" for group in groups}) == len(groups)
assert dict(contract.REWARD_VALUES) == {
    "exact": 1.0,
    "same_ab": 0.4,
    "same_a": 0.15,
    "same_domain": 0.01,
    "other_domain": 0.0,
}
print({"config_equal": True, "groups": len(groups), "offline_calibration": False})
PY
