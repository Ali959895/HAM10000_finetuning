#!/bin/bash
#SBATCH --job-name=blip2_opt27b_grid
#SBATCH --account=def-wassim
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --array=0-17
#SBATCH --output=/scratch/ali95/logs/blip2_grid_%A_%a.out
#SBATCH --error=/scratch/ali95/logs/blip2_grid_%A_%a.err

module --force purge
module load StdEnv/2023
module load gcc
module load opencv

source /home/ali95/env/bin/activate

export WANDB_MODE=offline
export WANDB_DISABLE_SERVICE=true
export HF_HOME=/scratch/ali95/hf_cache

LR_LIST=(1e-4 5e-5 3e-5)
LORA_R_LIST=(8 16 32)
LORA_TARGETS_LIST=("q_proj,v_proj" "q_proj,k_proj,v_proj,out_proj")

IDX=${SLURM_ARRAY_TASK_ID}
NUM_LR=3
NUM_R=3

LR=${LR_LIST[$((IDX % NUM_LR))]}
LORA_R=${LORA_R_LIST[$(((IDX / NUM_LR) % NUM_R))]}
LORA_TARGETS=${LORA_TARGETS_LIST[$((IDX / (NUM_LR * NUM_R)))]}

OUTDIR=/scratch/ali95/blip2_runs/grid/${SLURM_JOB_ID}_${IDX}

echo "================================"
echo "LR=${LR}"
echo "LORA_R=${LORA_R}"
echo "LORA_TARGETS=${LORA_TARGETS}"
echo "OUTDIR=${OUTDIR}"
echo "================================"

python blip2_opt_eva02_l14_coco_v2.py \
  --lavis_repo /home/ali95/LAVIS \
  --ann_train /scratch/ali95/coco/coco/annotations/captions_train2017.json \
  --ann_val /scratch/ali95/coco/coco/annotations/captions_val2017.json \
  --images_train /scratch/ali95/coco/coco/images/train2017 \
  --images_val /scratch/ali95/coco/coco/images/val2017 \
  --output_dir ${OUTDIR} \
  --model_type pretrain_opt2.7b \
  --epochs 50 \
  --batch_size 16 \
  --accum_steps 4 \
  --num_workers 4 \
  --lr ${LR} \
  --use_lora \
  --lora_r ${LORA_R} \
  --lora_targets ${LORA_TARGETS} \
  --bf16 \
  --use_wandb \
  --wandb_project blip2_coco
