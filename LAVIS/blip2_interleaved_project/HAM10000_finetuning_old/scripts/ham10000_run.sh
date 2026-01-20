#!/bin/bash
#SBATCH --job-name=ham10000_blip2
#SBATCH --account=def-wassim
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=/scratch/ali95/logs/%x_%j.out
#SBATCH --error=/scratch/ali95/logs/%x_%j.err

set -euo pipefail


# ====== Paths ======
PROJ=/home/$USER/LAVIS/blip2_interleaved_project/HAM10000_finetuning
ENV_DIR=/home/$USER/LAVIS/blip2_interleaved_project/blip2int_env

CONFIG=${1:-${PROJ}/configs/ham10000_finetune.yaml}
MODE=${2:-train_multiclass}
SPLITS=/home/ali95/LAVIS/datasets/HAM10000/splits
# ====== Modules / env ======
module purge
module load python/3.11 gcc/12.3.0 cuda/12.2 2>/dev/null || true
module load gcc opencv
source ${ENV_DIR}/bin/activate

mkdir -p /scratch/$USER/logs

# ====== Caches (avoid $HOME quota issues) ======
export HF_HOME=/scratch/$USER/hf_home
export TRANSFORMERS_CACHE=$HF_HOME
#export TORCH_HOME=/scratch/$USER/torch_home
export TOKENIZERS_PARALLELISM=false
export TORCH_HOME=/home/ali95/.cache/torch  
# python path
PROJECT_DIR=/home/ali95/LAVIS/blip2_interleaved_project
LAVIS_DIR=/home/ali95/LAVIS

export PYTHONPATH="$LAVIS_DIR:$PROJECT_DIR/src:${PYTHONPATH:-}"

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
#python src/run.py -c "$CONFIG" --mode "$MODE"

# 1) (one-time) make lesion-level split
#mkdir -p "${SPLITS}"
#python ${PROJ}/scripts/ham10000_split_by_lesion.py \
#  --meta_csv "${META}" \
#  --images_dir "${IMAGES}" \
#  --out_dir "${SPLITS}" \
#  --seed 42

# 2) run zero-shot on VAL (prompt ensembling)
#--mode zeroshot_multiclass

#--mode train_multiclass

#--mode eval_multiclass
python ${PROJ}/src/run.py \
  -c ${PROJ}/configs/ham10000_finetune.yaml \
  --mode train_multiclass

# For finetune:
# python ${PROJ}/src/run.py -c ${PROJ}/configs/ham10000_finetune.yaml --mode train_multiclass