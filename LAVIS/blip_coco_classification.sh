#!/bin/bash
#SBATCH --job-name=blip_coco_classification
#SBATCH --account=def-wassim
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/ali95/logs/blip_coco_classification_%j.out
#SBATCH --error=/scratch/ali95/logs/blip_coco_classification_%j.err

unset LD_LIBRARY_PATH
unset LD_PRELOAD
unset PYTHONPATH

module --force purge
module load StdEnv/2023 gcc/12.3 python/3.11 cuda/12.2 opencv

source /home/ali95/blip_env/bin/activate
cd /home/ali95/LAVIS

export PYTHONNOUSERSITE=1
# HuggingFace offline + cache
export HF_HOME=/scratch/ali95/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/ali95/.cache/huggingface
export HF_HUB_CACHE=/scratch/ali95/.cache/huggingface/hub
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
#torchrun --nproc_per_node=1 train.py \
#  --cfg-path projects/blip/coco_classification.yaml
export XDG_CACHE_HOME=/scratch/ali95/.cache
export TORCH_HOME=/scratch/ali95/.cache/torch
mkdir -p $TORCH_HOME/hub/checkpoints
export TORCH_HOME=/scratch/ali95/.cache/torch
export HF_HOME=/scratch/ali95/.cache/huggingface
export TRANSFORMERS_CACHE=/scratch/ali95/.cache/huggingface
mkdir -p $TORCH_HOME/hub/checkpoints $HF_HOME


echo "start training"
# Run with torchrun from the SAME venv
#python /home/ali95/LAVIS/train.py \
#  --cfg-path projects/blip/coco_classification.yaml
torchrun \
  --nproc_per_node=1 \
  train.py \
  --cfg-path projects/blip/coco_classification.yaml
