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

# Clean modules (Narval-safe)
module --force purge
module load StdEnv/2023
module load cuda/12.2
module load cudnn/9.2.1.18

# Activate your env
source /home/ali95/env/bin/activate

# Offline Hugging Face (critical)
export HF_HOME=/scratch/ali95/hf_home
export HF_HOME=/scratch/ali95/.cache/huggingface
export TRANSFORMERS_CACHE=$HF_HOME
export TIMM_HOME=/scratch/ali95/.cache/timm
export TORCH_HOME=/scratch/ali95/.cache/torch
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

# ---- W&B OFFLINE ----
# --- W&B OFFLINE MODE ---
export WANDB_MODE=offline
export WANDB_SILENT=true
export WANDB_DIR=/scratch/ali95/wandb
export WANDB_CACHE_DIR=/scratch/ali95/wandb/cache
export WANDB_CONFIG_DIR=/scratch/ali95/wandb/config
export WANDB_AGENT_DISABLE_FLAPPING=true
mkdir -p /scratch/ali95/blip2_logs /scratch/ali95/wandb

# Speed + stability
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
cd /home/ali95/LAVIS

# Paths (EDIT THESE)
LAVIS_REPO=/home/ali95/LAVIS

COCO_ROOT=/scratch/ali95/coco/coco/ \
ANN_TRAIN=${COCO_ROOT}/annotations/captions_train2017.json
ANN_VAL=${COCO_ROOT}/annotations/captions_val2017.json
IM_TRAIN=${COCO_ROOT}/images/train2017
IM_VAL=${COCO_ROOT}/images/val2017

OUTDIR=/scratch/ali95/blip2_runs/blip2_opt2.7b_eva02l14_coco
echo "Starting W&B agent (offline)..."

echo "=== Creating sweep ==="
SWEEP_OUTPUT=$(wandb sweep sweep_lora_opt27b.yaml 2>&1)

echo "$SWEEP_OUTPUT"
# Extract sweep ID safely
SWEEP_ID=$(echo "$SWEEP_OUTPUT" \
  | grep -oE 'with ID: [a-z0-9]+' \
  | awk '{print $3}')
  
if [ -z "$SWEEP_ID" ]; then
  echo "? Failed to extract sweep ID"
  echo "$SWEEP_OUTPUT"
  exit 1
fi

echo "Sweep ID = $SWEEP_ID"


echo "=== Starting agent ==="
wandb agent ali-algumaei-concordia-university/LAVIS/$SWEEP_ID
echo "Done at $(date)"

#python /home/ali95/LAVIS/blip2_opt_eva02_l14_coco_v2.py \
#  --lavis_repo ${LAVIS_REPO} \
#  --coco_root ${COCO_ROOT} \
#  --ann_train ${ANN_TRAIN} \
#  --ann_val ${ANN_VAL} \
#  --images_train ${IM_TRAIN} \
#  --images_val ${IM_VAL} \
#  --output_dir ${OUTDIR} \
#  --model_type pretrain_opt2.7b \
#  --epochs 50 \
#  --batch_size 16 \
#  --accum_steps 4 \
#  --num_workers 4 \
#  --lr 5e-5 \
#  --weight_decay 0.0 \
#  --grad_clip 1.0 \
#  --bf16 \
#  --use_lora \
#  --lora_r 16 \
#  --lora_alpha 32 \
#  --lora_dropout 0.05 \
#  --lora_targets q_proj,k_proj,v_proj,out_proj \
#  --use_wandb \
#  --wandb_offline \
#  --wandb_project blip2_coco \
#  --wandb_run_name blip2_opt27b_eva02l14_lora \
#  --log_every 50 \
#  --val_every 500 \
#  --ckpt_every 1000

echo "Done at $(date)"
