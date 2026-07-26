#!/usr/bin/env bash
set -euo pipefail

source "$(dirname "$0")/common.sh"
cd "${ROOT}"
PYTHONDONTWRITEBYTECODE=1 "${PYTHON}" -m unittest discover \
  -s tests/dapo_anchor_v2_2 -v
