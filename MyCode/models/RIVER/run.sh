#!/usr/bin/env bash
set -euo pipefail
PROJECT_ROOT=$(cd $(dirname ${BASH_SOURCE[0]})/../.. && pwd)
PYTHON=${PYTHON:-python3}
cd ${PROJECT_ROOT}
exec ${PYTHON} models/RIVER/benchmark_adapter.py "$@"
