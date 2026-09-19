#!/bin/bash -l




















export MASTER_PORT=25102
export WORLD_SIZE=2



echo "NODELIST="${SLURM_NODELIST}
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr
echo "MASTER_ADDR="$MASTER_ADDR

module load timm/1.0.8-foss-2023a-CUDA-12.1.1
module load wandb/0.16.1-GCC-12.3.0
module load einops/0.7.0-GCCcore-12.3.0
module load h5py/3.9.0-foss-2023a


export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


echo "Python executable: $(which python)"
echo "PYTHONPATH head: $(echo $PYTHONPATH | tr ':' '\n' | head -n 3)"


srun --mpi=pmi2 --export=ALL,PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(which python) ddp_finetune_eeg.py --challenge 'challenge1' --model 'vit_small_patch16' --output_dir 'outputs/eeg_foundation_2025' --vit_pretrained_model_dir 'checkpoints/eeg_foundation_2025/checkpoint.pth' --use_lora --lora_last_n 0 --lora_r 8 --lora_wd_zero --lora_lr_scale 1.0 --lora_no_lrd
