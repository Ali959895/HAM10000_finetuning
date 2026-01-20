#!/bin/bash
#SBATCH --job-name=ham10000_v3_100
#SBATCH --account=def-wassim
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=5-00:00:00
#SBATCH --output=/scratch/ali95/logs/%x_%j.out
#SBATCH --error=/scratch/ali95/logs/%x_%j.err

set -euo pipefail

# ====== Modules / env ======
ENV_DIR=/home/$USER/LAVIS/blip2_interleaved_project/blip2int_env
HOME=/home/ali95
module purge
module load python/3.11 gcc/12.3.0 cuda/12.2 2>/dev/null || true
module load gcc opencv

# Activate your env (edit if needed)
# source /home/$USER/blip2int_env/bin/activate
# or: source /home/$USER/LAVIS/blip2int_env/bin/activate

export WANDB_DIR=/scratch/$USER/wandb
mkdir -p "$WANDB_DIR" /scratch/$USER/logs

# ====== Caches (avoid $HOME quota issues) ======
export HF_HOME=/scratch/$USER/hf_home
export TRANSFORMERS_CACHE=$HF_HOME
#export TORCH_HOME=/scratch/$USER/torch_home
export TOKENIZERS_PARALLELISM=false
export TORCH_HOME=/scratch/ali95/.cache/torch
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
# python path
PROJECT_DIR=/home/ali95/LAVIS/blip2_interleaved_project
LAVIS_DIR=/home/ali95/LAVIS
export PYTHONPATH="$LAVIS_DIR:$PROJECT_DIR/src:${PYTHONPATH:-}"
source ${ENV_DIR}/bin/activate
python -c "import lavis; print('lavis ok')"
# ====== W&B OFFLINE (HPC-safe) ======
export WANDB_MODE=offline
export WANDB_DISABLE_SERVICE=true
export WANDB_START_METHOD=thread
export WANDB_SILENT=true
export WANDB_INIT_TIMEOUT=300


# Train one model:
PROJ=/scratch/ali95/blip2_interleaved_project/HAM10000_finetuning_v3
#export PYTHONPATH=$PROJ/src:$HOME/LAVIS
#python ${PROJ}/src/run.py \
#  -c ${PROJ}/configs/ham10000_finetune.yaml \
#  --mode train_multiclass

# Or run full benchmark (BLIP / BLIP2 / CLIP):
python ${PROJ}/src/run.py -c ${PROJ}/configs/ham10000_finetune_v3.yaml --mode benchmark_multiclass

# Or eval:
# python ${PROJ}/src/run.py -c ${PROJ}/configs/ham10000_finetune_v3.yaml --mode eval_multiclass
