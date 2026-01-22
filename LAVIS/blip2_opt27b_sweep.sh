#!/bin/bash
#SBATCH --job-name=blip2_opt27b_sweep
#SBATCH --account=def-wassim
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/ali95/logs/blip2_opt27b_sweep_%j.out
#SBATCH --error=/scratch/ali95/logs/blip2_opt27b_sweep_%j.err

# -------------------------------
# Modules
# -------------------------------
module --force purge
module load StdEnv/2023
module load gcc/12.3
module load cuda/12.2

# -------------------------------
# Activate virtual environment
# -------------------------------
source /home/ali95/env/bin/activate

# -------------------------------
# FORCE W&B OFFLINE (CRITICAL)
# -------------------------------
export WANDB_MODE=offline
export WANDB_DIR=/scratch/ali95/wandb
export WANDB_CACHE_DIR=/scratch/ali95/wandb/cache
export WANDB_CONFIG_DIR=/scratch/ali95/wandb/config
export WANDB_DISABLE_SERVICE=true
export WANDB_SILENT=true

mkdir -p /scratch/ali95/wandb

# -------------------------------
# Debug info
# -------------------------------
echo "Python: $(which python)"
echo "W&B mode: $WANDB_MODE"
nvidia-smi

# -------------------------------
# Run sweep agent OR single run
# -------------------------------

# OPTION A: SWEEP
wandb agent ali-algumaei-concordia-university/LAVIS/kj4aaj5f

# OPTION B: SINGLE RUN (comment sweep line above and use this)
# python blip2_opt_eva02_l14_coco.py \
#   --epochs 15 \
#   --batch_size 8 \
#   --lr 1e-4 \
#   --vision_depth 10
