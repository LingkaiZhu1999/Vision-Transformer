#!/bin/bash
#SBATCH --time=20:00:00
#SBATCH --mem=300G
#SBATCH --cpus-per-gpu=40
#SBATCH --gpus=2
#SBATCH --partition=gpu-h200-141g-ellis

# module load scicomp-python-env
source .venv/bin/activate
module load triton/2025.1-gcc
module load gcc

# 1. Find the hostname of the first node allocated to the job
MASTER_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-2}
WORKERS_PER_GPU=12
TOTAL_WORKERS=$((GPUS_PER_NODE * WORKERS_PER_GPU))

# Usage:
#   sbatch run.sh pretrain
#   sbatch run.sh finetune
MODE=${1:-finetune}

if [[ "$MODE" == "pretrain" ]]; then
  python main.py /scratch/shareddata/dldata/imagenet-1k-wds/imagenet-1k-wds/ \
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
    --dist-backend nccl \
    --compile \
    --bf16 \
    --use_zero
elif [[ "$MODE" == "finetune" ]]; then
  python main.py /scratch/shareddata/dldata/imagenet-1k-wds/imagenet-1k-wds/ \
    --batch-size 512 \
    --workers "$TOTAL_WORKERS" \
    --epochs 8 \
    --lr 3e-3 \
    --finetune-checkpoint ./checkpoint.pth.tar \
    --multiprocessing-distributed \
    --world-size $SLURM_NNODES \
    --rank $SLURM_NODEID \
    --dist-url "tcp://$MASTER_NODE:10001" \
    --dist-backend nccl \
    --bf16 \
    --use_zero
else
  echo "Invalid mode: $MODE"
  echo "Use one of: pretrain, finetune"
  exit 1
fi

