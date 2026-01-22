#!/bin/bash
#SBATCH --job-name=blip_coco_cls
#SBATCH --account=def-wassim
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --output=/scratch/ali95/logs/blip_coco_cls_%j.out
#SBATCH --error=/scratch/ali95/logs/blip_coco_cls_%j.err

module --force purge
module load StdEnv/2023 cuda/12.2 cudnn/9.2.1.18
source /home/ali95/env/bin/activate

source ~/env/bin/activate
export WORLD_SIZE=1
export RANK=0
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export CUDA_VISIBLE_DEVICES=0
export LOCAL_RANK=0
export RANK=0
export WORLD_SIZE=1

export NCCL_DEBUG=INFO
export WANDB_MODE=offline
export WANDB_DIR=/scratch/ali95/wandb
export PYTHONPATH=$PWD
export WANDB_MODE=offline
export WANDB_DIR=/scratch/ali95/wandb
export HF_HOME=/scratch/ali95/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export TORCH_HOME=/home/ali95/.cache/torch
export HF_DATASETS_OFFLINE=1
# W&B offline (critical)
export WANDB_MODE=offline
export WANDB_DIR=/scratch/ali95/wandb

cd /home/ali95/LAVIS

torchrun \
  --nproc_per_node=1 \
  train.py \
  --cfg-path projects/blip/coco_classification.yaml

