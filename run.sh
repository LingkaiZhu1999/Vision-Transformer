#!/bin/bash
#SBATCH --time=60:00:00
#SBATCH --mem=180G
#SBATCH --cpus-per-gpu=40
#SBATCH --gpus=2
#SBATCH --ntasks-per-node=1  # <--- Ensures 1 Python command is run per node
#SBATCH --partition=gpu-h200-141g-ellis

module load scicomp-python-env

# 1. Find the hostname of the first node allocated to the job
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)

# 2. Use srun and pass SLURM variables to Python
srun python main.py \
  "/scratch/shareddata/dldata/imagenet-1k-wds/imagenet-1k-wds/" \
  --batch-size 4096 \
  --workers 16 \
  --epochs 300 \
  --weight-decay 0.3 \
  --lr 3e-3 \
  --min_lr 1e-6 \
  --t_warm_up 10000 \
  --dist-url "tcp://$MASTER_NODE:10001" \
  --multiprocessing-distributed \
  --world-size $SLURM_NNODES \
  --rank $SLURM_NODEID \
  --dist-backend 'nccl' \
  --compile \
  --bf16 \
  --use_zero
