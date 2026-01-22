#!/bin/bash
#SBATCH --job-name=blip_vitl
#SBATCH --account=def-wassim
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100_4g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1-12:00:00
#SBATCH --output=/scratch/ali95/blip2_vitl/%x_%j.out
#SBATCH --error=/scratch/ali95/blip2_vitl/%x_%j.err

set -euo pipefail

echo "=== Job info ==="
echo "JobID:  ${SLURM_JOB_ID}"
echo "Node:   $(hostname)"
echo "GPUs:   ${SLURM_GPUS_ON_NODE:-1}"
echo "Time:   $(date)"
echo "==============="

# -------------------------
# Modules
# -------------------------
module --force purge
module load StdEnv/2023
module load cuda/12.2
module load cudnn/9.2.1.18

# -------------------------
# Python env
# -------------------------
source /home/ali95/env/bin/activate

# -------------------------
# Offline HuggingFace
# -------------------------
export HF_HOME=/scratch/ali95/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TIMM_HOME=/scratch/ali95/.cache/timm
export TORCH_HOME=/scratch/ali95/.cache/torch
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# -------------------------
# W&B offline
# -------------------------
export WANDB_MODE=offline
export WANDB_DIR=/scratch/ali95/wandb
export WANDB_START_METHOD=thread
mkdir -p /scratch/ali95/blip2_logs /scratch/ali95/wandb

# -------------------------
# Stability (MIG)
# -------------------------
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export TORCH_CUDA_ALLOC_CONF=max_split_size_mb:128,expandable_segments:True

# -------------------------
# Paths
# -------------------------
export LAVIS_REPO=/home/ali95/LAVIS

export COCO_ROOT=/scratch/ali95/coco/coco
export ANN_TRAIN=${COCO_ROOT}/annotations/instances_train2017.json
export ANN_VAL=${COCO_ROOT}/annotations/instances_val2017.json
export IM_TRAIN=${COCO_ROOT}/images/train2017
export IM_VAL=${COCO_ROOT}/images/val2017

#cd /scratch/ali95/blip2_coco

# -------------------------
# Model config (ViT-L recommended)
# -------------------------
MODEL_NAME=blip2
MODEL_TYPE=pretrain_vitl
SWEEP_COUNT=10

bash /home/ali95/LAVIS/launch_sweep.sh ${MODEL_NAME} ${MODEL_TYPE} ${SWEEP_COUNT}

