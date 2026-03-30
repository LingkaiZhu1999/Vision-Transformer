#!/bin/bash
#SBATCH --time=80:00:00
#SBATCH --mem=200G
#SBATCH --cpus-per-gpu=40
#SBATCH --gpus=2
#SBATCH --partition=gpu-b300-288g-ellis

# module load scicomp-python-env
source .venv/bin/activate
module load triton/2025.1-gcc
module load gcc

# 1. Find the hostname of the first node allocated to the job
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-2}
WORKERS_PER_GPU=12
TOTAL_WORKERS=$((GPUS_PER_NODE * WORKERS_PER_GPU))

# 2. Use srun and pass SLURM variables to Python
srun python main.py \
  "/scratch/shareddata/dldata/imagenet-1k-wds/imagenet-1k-wds/" \
  --batch-size 4096 \
  --workers "$TOTAL_WORKERS" \
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
