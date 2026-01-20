#!/usr/bin/env bash
#SBATCH --account=def-wassim
#SBATCH --job-name=coco_zs_blip2
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=6-00:00:00
#SBATCH --output=/scratch/ali95/logs/%x_%j.out
#SBATCH --error=/scratch/ali95/logs/%x_%j.err

set -euo pipefail

# ---- Modules / env ----
module load python/3.11
module load cuda/12.2
module load gcc opencv

PROJECT_DIR="/home/ali95/LAVIS/blip2_interleaved_project"
source "${PROJECT_DIR}/blip2int_env/bin/activate"

mkdir -p /scratch/$USER/logs
cd "${PROJECT_DIR}"
LAVIS_DIR=/home/ali95/LAVIS

export PYTHONPATH="$LAVIS_DIR:$PROJECT_DIR/src:${PYTHONPATH:-}"
# ---- Common cache setup (your file) ----
source "${PROJECT_DIR}/scripts/narval_common.sh"

# ---- Caches (ensure LAVIS finds checkpoints in offline mode) ----
export TORCH_HOME="${TORCH_HOME:-/scratch/$USER/torch_home}"
export HF_HOME="${HF_HOME:-/scratch/$USER/hf_home}"
export WANDB_MODE="${WANDB_MODE:-offline}"
mkdir -p "${TORCH_HOME}/hub/checkpoints" "${HF_HOME}"
source /home/ali95/LAVIS/blip2_interleaved_project/scripts/narval_common.sh

export TORCH_HOME=/scratch/ali95/vlm_cache/torch
export HF_HOME=/scratch/ali95/vlm_cache/hf

export COCO_TRAIN_IMAGES_DIR=/scratch/ali95/coco/coco/images/train2017
export COCO_TRAIN_ANN_JSON=/scratch/ali95/coco/coco/annotations/instances_train2017.json
export COCO_VAL_IMAGES_DIR=/scratch/ali95/coco/coco/images/val2017
export COCO_VAL_ANN_JSON=/scratch/ali95/coco/coco/annotations/instances_val2017.json

# ---- COCO paths (defaults; can be overridden via sbatch --export=ALL,COCO_ROOT=...) ----
COCO_ROOT="${COCO_ROOT:-/scratch/ali95/coco/coco}"
COCO_TRAIN_IMAGES_DIR="${COCO_TRAIN_IMAGES_DIR:-${COCO_ROOT}/images/train2017}"
COCO_TRAIN_ANN_JSON="${COCO_TRAIN_ANN_JSON:-${COCO_ROOT}/annotations/instances_train2017.json}"
COCO_VAL_IMAGES_DIR="${COCO_VAL_IMAGES_DIR:-${COCO_ROOT}/images/val2017}"
COCO_VAL_ANN_JSON="${COCO_VAL_ANN_JSON:-${COCO_ROOT}/annotations/instances_val2017.json}"

# ---- Validate paths early (clear error if wrong) ----
for p in "$COCO_TRAIN_IMAGES_DIR" "$COCO_VAL_IMAGES_DIR"; do
  [[ -d "$p" ]] || { echo "[ERROR] Missing directory: $p" >&2; exit 2; }
done
for p in "$COCO_TRAIN_ANN_JSON" "$COCO_VAL_ANN_JSON"; do
  [[ -f "$p" ]] || { echo "[ERROR] Missing file: $p" >&2; exit 2; }
done

# ---- Check required checkpoint for BLIP2 zeroshot in this project ----
CKPT_REQ="${TORCH_HOME}/hub/checkpoints/blip2_pretrained.pth"
if [[ ! -f "$CKPT_REQ" ]]; then
  echo "[ERROR] Missing checkpoint: $CKPT_REQ" >&2
  echo "Place blip2_pretrained.pth in: ${TORCH_HOME}/hub/checkpoints/" >&2
  exit 3
fi

# ---- Run dir ----
RUNS_DIR=/scratch/ali95/blip2_interleaved_project/runs
export RUN_DIR="${RUNS_DIR}/coco_zeroshot_trainval_${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"
echo "[INFO] RUN_DIR=$RUN_DIR"


CONFIG_PATH="${PROJECT_DIR}/configs/coco_zeroshot_trainval.yaml"

echo "[INFO] RUN_DIR=${RUN_DIR}"
echo "[INFO] Using config: ${CONFIG_PATH}"
echo "[INFO] TORCH_HOME=${TORCH_HOME}"

run_split () {
  local split="$1"
  local images_dir="$2"
  local ann_json="$3"
  local out_dir="${RUN_DIR}/${split}"

  mkdir -p "${out_dir}"
  echo "[INFO] ===== Running zeroshot split=${split} ====="
  echo "[INFO] images_dir=${images_dir}"
  echo "[INFO] ann_json=${ann_json}"
  echo "[INFO] out_dir=${out_dir}"

  python src/run.py \
    --config "${CONFIG_PATH}" \
    --mode zeroshot \
    --run_dir "${out_dir}" \
    --options \
      "data.num_workers=${SLURM_CPUS_PER_TASK}" \
      "data.coco.images_dir=${images_dir}" \
      "data.coco.ann_json=${ann_json}"
}


# Evaluate both splits (train + val) in one job
run_split train "${COCO_TRAIN_IMAGES_DIR}" "${COCO_TRAIN_ANN_JSON}"
run_split val   "${COCO_VAL_IMAGES_DIR}"   "${COCO_VAL_ANN_JSON}"

# Merge the two final_metrics.json into one file for convenience
python - <<'PY'
import json, os, pathlib
run_dir = pathlib.Path(os.environ["RUN_DIR"])
out = {"train": None, "val": None}
for split in ["train","val"]:
    p = run_dir/split/"final_metrics.json"
    if p.exists():
        out[split] = json.loads(p.read_text())
print("[INFO] Combined metrics:", out)
(run_dir/"final_metrics_trainval.json").write_text(json.dumps(out, indent=2))
print("[INFO] Wrote:", run_dir/"final_metrics_trainval.json")
PY
