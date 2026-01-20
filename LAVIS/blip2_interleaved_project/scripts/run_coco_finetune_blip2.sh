#!/usr/bin/env bash
set -euo pipefail

# COCO multi-label FINE-TUNING using BLIP-2 features + trainable classifier head.
#
# Required:
#   COCO_TRAIN_IMAGES_DIR=/path/to/coco/images/train2017
#   COCO_TRAIN_ANN_JSON=/path/to/coco/annotations/instances_train2017.json
#   COCO_VAL_IMAGES_DIR=/path/to/coco/images/val2017
#   COCO_VAL_ANN_JSON=/path/to/coco/annotations/instances_val2017.json
#
# Hyperparams (override):
#   LR=1e-4 BS=32 WD=0.05 OPT=adamw EPOCHS=10
#
# Usage:
#   bash scripts/run_coco_finetune_blip2.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${ROOT_DIR}/scripts/narval_common.sh"

: "${COCO_TRAIN_IMAGES_DIR:?Must set COCO_TRAIN_IMAGES_DIR}"
: "${COCO_TRAIN_ANN_JSON:?Must set COCO_TRAIN_ANN_JSON}"
: "${COCO_VAL_IMAGES_DIR:?Must set COCO_VAL_IMAGES_DIR}"
: "${COCO_VAL_ANN_JSON:?Must set COCO_VAL_ANN_JSON}"

RUNS_DIR="${RUNS_DIR:-${SCRATCH_DIR}/blip2_interleaved_project/runs}"
mkdir -p "${RUNS_DIR}"

LR="${LR:-1e-4}"
BS="${BS:-32}"
WD="${WD:-0.05}"
OPT="${OPT:-adamw}"
EPOCHS="${EPOCHS:-10}"

python "${ROOT_DIR}/src/run.py" \
  --config "${ROOT_DIR}/configs/coco_finetune.yaml" \
  --mode train \
  --run_dir "${RUNS_DIR}/coco_finetune_blip2" \
  --options \
    "exp_name=coco_finetune_blip2" \
    "optim.name=${OPT}" \
    "optim.lr=${LR}" \
    "optim.weight_decay=${WD}" \
    "train.batch_size=${BS}" \
    "train.epochs=${EPOCHS}"
