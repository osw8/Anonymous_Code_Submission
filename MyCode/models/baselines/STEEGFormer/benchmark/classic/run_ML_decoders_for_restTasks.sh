#!/bin/bash -l

















CONDA_ROOT="${CONDA_ROOT:-$PWD/.conda}"
export PATH="$CONDA_ROOT/bin:$PATH"
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV_NAME:-env1}"


downstream_tasks=(error)
evaluation_schemes=(population leave-one-out-finetuning per-subject)
models=(xdawn_lda xdawncov_mdm xdawncov_ts_svm erpcov_mdm dcpm)

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


python run_ML_decoders_for_restTasks.py \
  --model "$model" \
  --downstream_task "$downstream_task" \
  --evaluation_scheme "$evaluation_scheme" \
  --seed 3407
