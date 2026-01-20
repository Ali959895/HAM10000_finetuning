#!/usr/bin/env bash
set -euo pipefail

# COCO multi-label ZERO-SHOT evaluation for all baselines (incl. interleaved BLIP-2).
#
# Required environment variables:
#   COCO_VAL_IMAGES_DIR=/path/to/coco/images/val2017
#   COCO_VAL_ANN_JSON=/path/to/coco/annotations/instances_val2017.json
#
# Optional:
#   RUNS_DIR=/scratch/$USER/blip2_interleaved_project/runs
#   EVAL_BS=32
#   SHOTS=0
#   BIOVIL_HF_MODEL_ID=...         (if using biovil baseline)
#   MAMMOCLIP_CKPT_PATH=/path/...  (if using mammoclip baseline)
#
# Usage:
#   bash scripts/run_coco_zeroshot_all.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# Narval-friendly environment defaults (caches, offline W&B, PYTHONPATH)
source "${ROOT_DIR}/scripts/narval_common.sh"

: "${COCO_VAL_IMAGES_DIR:?Must set COCO_VAL_IMAGES_DIR}"
: "${COCO_VAL_ANN_JSON:?Must set COCO_VAL_ANN_JSON}"

RUNS_DIR="${RUNS_DIR:-${SCRATCH_DIR}/blip2_interleaved_project/runs}"
mkdir -p "${RUNS_DIR}"

EVAL_BS="${EVAL_BS:-32}"
SHOTS="${SHOTS:-0}"

baselines=(clip blip blip2 blip2_interleaved_yesno biovil)

for baseline in "${baselines[@]}"; do
  run_dir="${RUNS_DIR}/coco_zeroshot_${baseline}"
  python "${ROOT_DIR}/src/run.py" \
    --config "${ROOT_DIR}/configs/coco_zeroshot.yaml" \
    --mode zeroshot \
    --run_dir "${run_dir}" \
    --options \
      "exp_name=coco_zeroshot_${baseline}" \
      "model.baseline=${baseline}" \
      "eval.batch_size=${EVAL_BS}" \
      "eval.shots=${SHOTS}"
done
