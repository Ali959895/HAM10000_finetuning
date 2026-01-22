#!/bin/bash
#SBATCH --job-name=ham10000_blip2
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
module purge
module load python/3.11 gcc/12.3.0 cuda/12.2 2>/dev/null || true
module load gcc opencv
source ${ENV_DIR}/bin/activate
mkdir -p /scratch/$USER/logs/ham_logs

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


#CFG=/scratch/$USER/ham_cfg/ham10000_finetune.yaml
#mkdir -p /scratch/$USER/ham_cfg
#cp -f $PROJ/configs/ham10000_finetune.yaml $CFG

#python $PROJ/src/run.py -c $CFG --mode train_multiclass
python ${PROJ}/src/run.py \
  -c ${PROJ}/configs/ham10000_finetune.yaml  \
  --mode train_multiclass
# Example evaluation (update checkpoint path after training):
# python $PROJ/src/run.py -c $CFG --mode eval_multiclass
