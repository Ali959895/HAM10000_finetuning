#!/bin/bash
#SBATCH --job-name=ham10000_v3_mix
#SBATCH --account=def-wassim
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --output=/scratch/ali95/logs/%x_%j.out
#SBATCH --error=/scratch/ali95/logs/%x_%j.err

set -euo pipefail

# ====== Modules ======
module purge
module load python/3.11 gcc/12.3.0 cuda/12.2 2>/dev/null || true
module load gcc opencv
# ====== Project paths ======
PROJ=/scratch/ali95/blip2_interleaved_project/HAM10000_finetuning_v3
CFG_BLIP2=${PROJ}/configs/ham10000_finetune_v3.yaml
CFG_BLIP=${PROJ}/configs/ham10000_blip.yaml

# ====== Environments ======
BLIP2_ENV=/home/$USER/LAVIS/blip2_interleaved_project/blip2int_env
BLIP_ENV=/home/ali95/blip_env

# ====== Caches (avoid $HOME quota issues) ======
export WANDB_DIR=/scratch/$USER/wandb
mkdir -p "$WANDB_DIR" /scratch/$USER/logs

export HF_HOME=/scratch/$USER/hf_home
export TRANSFORMERS_CACHE=$HF_HOME
export TORCH_HOME=/scratch/ali95/.cache/torch
export TOKENIZERS_PARALLELISM=false

# Offline mode (keep ON if compute nodes have no internet)
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ====== W&B OFFLINE (HPC-safe) ======
export WANDB_MODE=offline
export WANDB_DISABLE_SERVICE=true
export WANDB_START_METHOD=thread
export WANDB_SILENT=true
export WANDB_INIT_TIMEOUT=300

# ====== PYTHONPATH (LAVIS + your project code) ======
PROJECT_DIR=/home/ali95/LAVIS/blip2_interleaved_project
LAVIS_DIR=/home/ali95/LAVIS
export PYTHONPATH="$LAVIS_DIR:$PROJECT_DIR/src:${PYTHONPATH:-}"

echo "=== Run 1/2: BLIP2 + CLIP + others (blip2int_env) ==="
source ${BLIP2_ENV}/bin/activate
python -c "import transformers; print('Transformers (blip2int_env):', transformers.__version__)"
python -c "import lavis; print('lavis ok (blip2int_env)')"

python ${PROJ}/src/run.py -c ${CFG_BLIP2} --mode benchmark_multiclass
deactivate

echo "=== Run 2/2: BLIP only (blip_env) ==="
source ${BLIP_ENV}/bin/activate
python -c "import transformers; print('Transformers (blip_env):', transformers.__version__)"
python -c "import lavis; print('lavis ok (blip_env)')"

python ${PROJ}/src/run.py -c ${CFG_BLIP} --mode benchmark_multiclass
deactivate

echo "=== DONE ==="
