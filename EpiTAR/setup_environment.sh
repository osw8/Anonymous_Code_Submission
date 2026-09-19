#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
PYTHON_BIN=${PYTHON_BIN:-python3}
VENV_DIR=${VENV_DIR:-${PROJECT_ROOT}/.venv}

${PYTHON_BIN} -m venv ${VENV_DIR}
${VENV_DIR}/bin/python -m pip install --upgrade pip setuptools wheel
${VENV_DIR}/bin/python -m pip install -r ${PROJECT_ROOT}/requirement.txt
${VENV_DIR}/bin/python -m pip check

printf 'Environment ready: %s\n' ${VENV_DIR}
