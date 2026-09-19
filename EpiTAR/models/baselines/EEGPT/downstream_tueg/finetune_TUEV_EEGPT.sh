#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd $(dirname ${BASH_SOURCE[0]}) && pwd)
cd ${SCRIPT_DIR}

set -x


export MASTER_PORT=$((12000 + $RANDOM % 20000))


SPLIT=${SPLIT:-1}

N_NODES=${N_NODES:-1}
GPUS_PER_NODE=${GPUS_PER_NODE:-2}
SRUN_ARGS=${SRUN_ARGS:-""}
PY_ARGS=${@:2}


CUDA_VISIBLE_DEVICES=4,5 OMP_NUM_THREADS=1 python -m torch.distributed.run --nproc_per_node=${GPUS_PER_NODE} \
        --master_port ${MASTER_PORT} --nnodes=${N_NODES} --node_rank=0 --master_addr="localhost" \
        run_class_finetuning_EEGPT_change_tuev.py \
        --output_dir ./checkpoints_TUEV/finetune_tuev_eegpt/ \
        --log_dir ./log/finetune_tuev_eegpt \
        --model EEGPT \
        --finetune ../checkpoint/eegpt_mcae_58chs_4s_large4E.ckpt \
        --weight_decay 0.05 \
        --batch_size 400\
        --lr 5e-4 \
        --update_freq 1 \
        --warmup_epochs 5 \
        --epochs 30 \
        --layer_decay 0.65 \
        --drop_path 0.2 \
        --dist_eval \
        --save_ckpt_freq 5 \
        --disable_rel_pos_bias \
        --abs_pos_emb \
        --dataset TUEV\
        --enable_deepspeed \
        --seed 0
