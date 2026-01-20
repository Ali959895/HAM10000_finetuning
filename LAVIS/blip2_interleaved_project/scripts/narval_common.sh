#!/usr/bin/env bash
set -euo pipefail

# Common Narval / Compute Canada setup.
# Source this from sbatch scripts or interactive sessions.

# ---- Project root ----
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PROJECT_DIR
export PYTHONPATH="${PROJECT_DIR}/src:${PYTHONPATH:-}"
export TORCH_HOME=/scratch/ali95/vlm_cache/torch
export HF_HOME=/scratch/ali95/vlm_cache/hf

export COCO_TRAIN_IMAGES_DIR=/scratch/ali95/coco/coco/images/train2017
export COCO_TRAIN_ANN_JSON=/scratch/ali95/coco/coco/annotations/instances_train2017.json
export COCO_VAL_IMAGES_DIR=/scratch/ali95/coco/coco/images/val2017
export COCO_VAL_ANN_JSON=/scratch/ali95/coco/coco/annotations/instances_val2017.json

# ---- Caches on $SCRATCH (avoids $HOME quota + speeds up) ----
export SCRATCH_DIR="${SCRATCH:-${PROJECT_DIR}}"
export VLM_CACHE_ROOT="${VLM_CACHE_ROOT:-${SCRATCH_DIR}/vlm_cache}"
mkdir -p "${VLM_CACHE_ROOT}"

export HF_HOME="${HF_HOME:-${VLM_CACHE_ROOT}/hf}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HOME}/transformers}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCH_HOME="${TORCH_HOME:-${VLM_CACHE_ROOT}/torch}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-${VLM_CACHE_ROOT}/xdg}"
mkdir -p "${HF_HOME}" "${TRANSFORMERS_CACHE}" "${HUGGINGFACE_HUB_CACHE}" "${TORCH_HOME}" "${XDG_CACHE_HOME}"

# ---- W&B offline ----
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${VLM_CACHE_ROOT}/wandb}"
mkdir -p "${WANDB_DIR}"

# Recommended settings on clusters
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTHONUNBUFFERED=1

