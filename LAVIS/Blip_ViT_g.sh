#!/bin/bash
#SBATCH --job-name=blip2_vitg
#SBATCH --account=def-wassim
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --array=0-14
#SBATCH --output=/scratch/ali95/logs/%x_%A_%a.out
#SBATCH --error=/scratch/ali95/logs/%x_%A_%a.err

set -euo pipefail

module --force purge
module load StdEnv/2023 cuda/12.2 cudnn/9.2.1.18
source /home/ali95/env/bin/activate



export HF_HOME=/scratch/ali95/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/home/ali95/.cache/torch
export HF_DATASETS_OFFLINE=1
# W&B offline (critical)
export WANDB_MODE=offline
export WANDB_DIR=/scratch/ali95/wandb
export WANDB_CACHE_DIR=/scratch/ali95/wandb_cache
export WANDB_CONFIG_DIR=/scratch/ali95/wandb_config
export WANDB_PROJECT=coco-blip2-classification
export WANDB_ENTITY=ali-algumaei-concordia-university

mkdir -p /scratch/ali95/blip2_logs /scratch/ali95/wandb
# ----------------------------
# Hyperparameter grid
# ----------------------------
LR_LIST=(1e-6 5e-6 1e-5 5e-5 1e-4)
BS_LIST=(8 16 32)

IDX=${SLURM_ARRAY_TASK_ID}
LR=${LR_LIST[$((IDX / 3))]}
BS=${BS_LIST[$((IDX % 3))]}

echo "LR=${LR}, BS=${BS}"
# Paths (EDIT THESE)
LAVIS_REPO=/home/ali95/LAVIS

COCO_ROOT=/scratch/ali95/coco/coco/ \
ANN_TRAIN=${COCO_ROOT}/annotations/captions_train2017.json
ANN_VAL=${COCO_ROOT}/annotations/captions_val2017.json
IM_TRAIN=${COCO_ROOT}/train2017
IM_VAL=${COCO_ROOT}/val2017
# ----------------------------
# Run training (DIRECT)
# ----------------------------
python /home/ali95/LAVIS/train_coco.py \
  --model_name blip2_opt \
  --model_type pretrain_opt2.7b \
  --coco_root /scratch/ali95/coco/coco \
  --epochs 15 \
  --batch_size ${BS} \
  --lr ${LR} \
  --weight_decay 0.01 \
  --num_workers 4 \
  

