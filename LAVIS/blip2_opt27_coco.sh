#!/bin/bash
#SBATCH --account=def-wassim
#SBATCH --job-name=blip2_opt27b_coco
#SBATCH --output=/scratch/ali95/logs/%x_%j.out
#SBATCH --error=/scratch/ali95/logs/%x_%j.err
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00

set -euo pipefail

echo "=== Job info ==="
echo "JobID:  ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPUs:   ${SLURM_GPUS_ON_NODE:-1}"
echo "Time:   $(date)"
echo "==============="

############################
# Modules (Narval-safe)
############################
module --force purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2
module load cudnn/9.2.1.18

############################
# Python environment
############################
VENV_PATH=/home/ali95/env
if [ ! -d "$VENV_PATH" ]; then
  echo "❌ Virtualenv not found at $VENV_PATH"
  exit 1
fi

source $VENV_PATH/bin/activate
export DATA_DIR=/scratch/ali95

############################
# HuggingFace OFFLINE
############################
export HF_HOME=/scratch/ali95/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TIMM_HOME=/scratch/ali95/.cache/timm
export TORCH_HOME=/scratch/ali95/.cache/torch
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

############################
# W&B OFFLINE (CRITICAL)
############################
export WANDB_MODE=offline
export WANDB_DIR=/scratch/ali95/wandb
export WANDB_CACHE_DIR=/scratch/ali95/wandb/cache
export WANDB_CONFIG_DIR=/scratch/ali95/wandb/config

mkdir -p /scratch/ali95/logs /scratch/ali95/wandb

############################
# Performance flags
############################
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export COCO_ROOT=/scratch/ali95/coco/coco

############################
# Paths
############################
LAVIS_REPO=/home/ali95/LAVIS
COCO_ROOT=/scratch/ali95/coco/coco
OUTDIR=/scratch/ali95/blip2_runs/blip2_opt2.7b_eva02l14_coco

############################
# Run training
############################
cd ${LAVIS_REPO}

python blip2_opt27_coco.py \
  --coco_root ${COCO_ROOT} \
  --output_dir ${OUTDIR} \
  --epochs 15 \
  --batch_size 8 \
  --accum_steps 4 \
  --lr 1e-4 \
  --num_workers 4 \
  --bf16 \
  --use_lora \
  --lora_r 16 \
  --lora_alpha 32 \
  --lora_dropout 0.05 \
  --reduce_vision_blocks 10

echo "✅ Done at $(date)"
