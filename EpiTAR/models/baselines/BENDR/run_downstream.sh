#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
cd ${SCRIPT_DIR}

mkdir -p results


python3 downstream.py linear --random-init --results-filename "results/linear_random_init.xlsx"
python3 downstream.py BENDR --random-init --results-filename "results/BENDR_random_init.xlsx"







