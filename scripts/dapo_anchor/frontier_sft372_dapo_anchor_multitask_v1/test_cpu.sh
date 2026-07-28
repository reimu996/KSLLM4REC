#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"
cd "${ROOT}"

PYTHONDONTWRITEBYTECODE=1 "${PYTHON}" - "${CONFIG}" <<'PY'
import sys
from pathlib import Path

import yaml

from ksllm4rec_dapo_anchor_multitask_v1.config import validate_config

config = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8"))
validate_config(config)
PY
PYTHONDONTWRITEBYTECODE=1 "${PYTHON}" -m unittest discover \
  -s tests/dapo_anchor_multitask_v1 \
  -t . \
  -v

