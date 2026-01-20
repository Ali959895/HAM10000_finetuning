#!/bin/bash
#SBATCH --job-name=blip2_opt27b
#SBATCH --account=def-wassim
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/ali95/logs/blip2_opt27b_sweep_%j.out

module purge
module load StdEnv/2023
module load cuda/12.2
module load gcc

source /home/ali95/env/bin/activate

# -------------------------
# W&B setup (IMPORTANT)
# -------------------------
export WANDB_DIR=/scratch/ali95/wandb
export WANDB_MODE=offline            # keep offline on compute nodes
export WANDB_PROJECT=blip2_opt_coco
export WANDB_ENTITY=ali-algumaei-concordia-university

# Optional but recommended
export WANDB_CACHE_DIR=/scratch/ali95/wandb/cache

# -------------------------
# Data paths (fixed)
# -------------------------
export ANN_TRAIN=/scratch/ali95/coco/coco/annotations/captions_train2017.json
export ANN_VAL=/scratch/ali95/coco/coco/annotations/captions_val2017.json
export IMG_TRAIN=/scratch/ali95/coco/coco/train2017
export IMG_VAL=/scratch/ali95/coco/coco/val2017
export OUTDIR=/scratch/ali95/blip2_runs/blip2_opt2.7b_eva02l14_coco

mkdir -p $OUTDIR

# -------------------------
# Run W&B sweep agent
# -------------------------
# 🔴 REPLACE abc123 WITH YOUR REAL SWEEP ID
wandb agent ali-algumaei-concordia-university/blip2_opt_coco/zgae1vmu
