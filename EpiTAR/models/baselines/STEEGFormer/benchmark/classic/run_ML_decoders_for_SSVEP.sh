#!/bin/bash -l

















CONDA_ROOT="${CONDA_ROOT:-$PWD/.conda}"
export PATH="$CONDA_ROOT/bin:$PATH"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-env2}"


downstream_tasks=(binocular_ssvep)
evaluation_schemes=(per-subject)
models=(trca)

N1=${#downstream_tasks[@]}
N2=${#evaluation_schemes[@]}
N3=${#models[@]}


idx=$SLURM_ARRAY_TASK_ID
i=$(( idx / (N2 * N3) ))
rem=$(( idx % (N2 * N3) ))
j=$(( rem / N3 ))
k=$(( rem % N3 ))

downstream_task=${downstream_tasks[$i]}
evaluation_scheme=${evaluation_schemes[$j]}
model=${models[$k]}

echo "[$(date)] Combo #$idx: model=$model, task=$downstream_task, scheme=$evaluation_scheme"


python run_ML_decoders_for_SSVEP.py \
  --model "$model" \
  --downstream_task "$downstream_task" \
  --evaluation_scheme "$evaluation_scheme" \
  --seed 3407
