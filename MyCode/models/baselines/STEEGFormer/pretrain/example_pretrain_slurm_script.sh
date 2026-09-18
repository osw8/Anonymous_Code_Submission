#!/bin/bash -l




















export MASTER_PORT=13416
export WORLD_SIZE=2



echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR


module load Python/3.11.5-GCCcore-13.2.0



echo "Python executable: $(which python)"
export TORCH_DISTRIBUTED_DEBUG=DETAIL

srun --mpi=pmi2 --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(which python) ddp_train_eeg.py --model 'mae_vit_large_patch16' --resume 'latest' --output_dir './checkpoint/experiment5_large' --config_path './global_setting_pretrainDEBUG.json' --distributed true --device 'cuda' --batch_size 128 --accum_iter 4 --num_workers 8 --epochs 401
