#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

"$(dirname "$0")/test_cpu.sh"
"$(dirname "$0")/run_gates.sh"
"$(dirname "$0")/run_full.sh"
"$(dirname "$0")/verify.sh"
