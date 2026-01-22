#!/bin/bash
#SBATCH --job-name=blip2_coco_lora
#SBATCH --account=def-wassim
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=/scratch/ali95/logs/blip2_coco_lora_%j.out
#SBATCH --error=/scratch/ali95/logs/blip2_coco_lora_%j.err

set -euo pipefail

echo "=== Job info ==="
echo "JobID: ${SLURM_JOB_ID}"
echo "Node:  $(hostname)"
echo "GPUs:  ${SLURM_GPUS_ON_NODE:-1}"
echo "Time:  $(date)"
echo "==============="

# ---- Modules (StdEnv/2023 typical on Narval) ----
module load StdEnv/2023
module load cuda/12.2 2>/dev/null || true
module load cudnn 2>/dev/null || true

# ---- Activate your venv ----
# Update this path if your env lives elsewhere
source /home/ali95/env/bin/activate

# ---- Offline-safe settings (Compute Canada compute nodes have no internet) ----
export WANDB_MODE=offline
export WANDB_SILENT=true
export WANDB_START_METHOD=thread
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME=/scratch/ali95/hf_cache
export TRANSFORMERS_CACHE=/scratch/ali95/hf_cache/transformers
export HF_DATASETS_CACHE=/scratch/ali95/hf_cache/datasets

mkdir -p /scratch/ali95/blip2_logs
mkdir -p "$HF_HOME" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"
mkdir -p /scratch/ali95/runs

# ---- Optional: if you keep custom code in a repo, add it to PYTHONPATH ----
# export PYTHONPATH=/scratch/ali95/medblip/code/LAVIS-main:$PYTHONPATH

# ---- Run ----
CONFIG_PATH=${1:-coco_LORA.yaml}

echo "Running with config: ${CONFIG_PATH}"

srun python -u blip2_eva02_l14_coco_LORA.py \
  --config "${CONFIG_PATH}"
