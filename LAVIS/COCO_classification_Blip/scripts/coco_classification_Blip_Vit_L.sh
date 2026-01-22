#!/bin/bash
#SBATCH --account=def-wassim
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --job-name=Blip_vitl_coco
#SBATCH --output=/scratch/ali95/logs/%x_%j.out
#SBATCH --error=/scratch/ali95/logs/%x_%j.err

set -euo pipefail

module load python/3.11
module load cuda/12.2  # if needed on your cluster
module load gcc opencv


source /home/ali95/blip_env/bin/activate

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH=/home/ali95/LAVIS:$PYTHONPATH  
export PYTHONPATH=/home/ali95/LAVIS:/home/ali95/LAVIS/COCO_classification_Blip/src:$PYTHONPATH
export HF_HOME=/scratch/ali95/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export OPENCLIP_CACHE_DIR=/scratch/ali95/open_clip_cache
export HF_HOME=/scratch/ali95/hf_cache
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export PYTHONPATH=$PWD/src:$PYTHONPATH

python -c "import lavis; print(lavis.__file__)"
# Debug sanity checks
# ========================
python - <<'EOF'
import cv2, lavis, open_clip
print("cv2:", cv2.__version__)
print("lavis OK")
print("open_clip OK")
EOF

# Launch training
# ========================
cd /home/ali95/LAVIS/COCO_classification_Blip
# torchrun env
export MASTER_ADDR=127.0.0.1
export MASTER_PORT=29500
export WORLD_SIZE=1
export RANK=0

cd /home/ali95/LAVIS/COCO_classification_Blip/
torchrun --nproc_per_node=1 src/train.py \
  -c configs/tasks/coco_multilabel_vitl.yaml \
  --options \
    model.pretrained_path=/scratch/ali95/hf_cache/open_clip_pytorch_model.safetensors \
    model.freeze_text=true
  --options run.iters_per_epoch=1000

#torchrun --nproc_per_node=1 src/train.py   -c configs/tasks/coco_multilabel_vitl.yaml   --options     run_cfg.output_dir=/scratch/ali95/blip_coco_cls     run_cfg.job_id=coco_clip_vitl_run     run.device=cuda run_cfg.device=cuda     run.num_workers=1 run_cfg.num_workers=1     run.lr_sched=linear_warmup_cosine_lr run_cfg.lr_sched=linear_warmup_cosine_lr     model.pretrained_path=/scratch/ali95/hf_cache/open_clip_pytorch_model.safetensors     model.freeze_text=true
#python src/train.py \
#  -c configs/tasks/coco_multilabel_vitl.yaml \
#  --options \
#    model.pretrained_path=/scratch/ali95/hf_cache/open_clip_pytorch_model.safetensors \
#    model.freeze_text=true
