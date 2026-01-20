#!/bin/bash
#SBATCH --job-name=ham10000_blip2
#SBATCH --account=def-wassim_g
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/%u/logs/%x_%j.out
#SBATCH --error=/scratch/%u/logs/%x_%j.err

set -euo pipefail

# ====== Paths ======
PROJ=/home/$USER/LAVIS/blip2_interleaved_project/HAM10000_fine_tuning
ENV_DIR=/home/$USER/LAVIS/blip2_interleaved_project/blip2int_env

CONFIG=${1:-${PROJ}/configs/ham10000_finetune.yaml}
MODE=${2:-train_multiclass}

# ====== Modules / env ======
module purge
module load python/3.11 gcc/12.3.0 cuda/12.2 2>/dev/null || true

source ${ENV_DIR}/bin/activate

mkdir -p /scratch/$USER/logs

# ====== Caches (avoid $HOME quota issues) ======
export HF_HOME=/scratch/$USER/hf_home
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=/scratch/$USER/torch_home
export TOKENIZERS_PARALLELISM=false

# ====== W&B OFFLINE (HPC-safe) ======
export WANDB_MODE=offline
export WANDB_DISABLE_SERVICE=true
export WANDB_START_METHOD=thread
export WANDB_SILENT=true
export WANDB_INIT_TIMEOUT=300
export WANDB_DIR=/scratch/$USER/wandb
mkdir -p "$WANDB_DIR"

# Helpful: avoid picking up previous W&B runs if multiple steps in same job
unset WANDB_RUN_ID || true

echo "[INFO] PROJ=$PROJ"
echo "[INFO] CONFIG=$CONFIG"
echo "[INFO] MODE=$MODE"
echo "[INFO] Python=$(which python)"

cd $PROJ
python src/run.py -c "$CONFIG" --mode "$MODE"
