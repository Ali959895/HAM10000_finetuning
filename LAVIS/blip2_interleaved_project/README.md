# BLIP-2 + Interleaved (Med-Flamingo-style) + LAVIS on Narval (Compute Canada)

This repository is designed to run cleanly on **Narval** (Slurm) with:
- **COCO multi-label**: zero-shot evaluation + fine-tuning
- **Interleaved BLIP-2 yes/no** scoring (Med-Flamingo-style interleaving of images/text)
- Fair baselines: **CLIP**, **BLIP (base)**, **BLIP-2**, **BioViL** (optional, if weights available)
- **W&B offline** logging
- **Best checkpoint** saved automatically

## 0) Copy project to Narval + unzip
From your PC:
```bash
scp vlm_blip2_interleaved_project.zip YOURUSER@narval.computecanada.ca:~/projects/
```

On Narval:
```bash
mkdir -p ~/projects/blip2_interleaved_project
cd ~/projects/blip2_interleaved_project
unzip ../vlm_blip2_interleaved_project.zip -d .
```

## 1) Environment + caches (important on Narval)
This project stores caches/logs under **$SCRATCH** by default to avoid $HOME quotas.
The helper script is:
```bash
source scripts/narval_common.sh
```

Recommended (adapt to your environment):
- Activate your existing env (example):
  ```bash
  # conda activate blip_env
  # OR: source ~/venvs/blip_env/bin/activate
  ```
- Ensure `lavis` is importable (installed or editable install from your LAVIS clone).

## 2) COCO zero-shot evaluation (all baselines)
Set COCO paths:
```bash
export COCO_VAL_IMAGES_DIR=/path/to/coco/val2017
export COCO_VAL_ANN_JSON=/path/to/coco/annotations/instances_val2017.json
```

Interactive:
```bash
bash scripts/run_coco_zeroshot_all.sh
```

Slurm:
```bash
sbatch --account=def-XXXX --export=ALL,COCO_VAL_IMAGES_DIR=...,COCO_VAL_ANN_JSON=... slurm/coco_zeroshot.sbatch
```

## 3) COCO fine-tuning (BLIP-2 classifier head)
Set COCO train/val paths:
```bash
export COCO_TRAIN_IMAGES_DIR=/path/to/coco/train2017
export COCO_TRAIN_ANN_JSON=/path/to/coco/annotations/instances_train2017.json
export COCO_VAL_IMAGES_DIR=/path/to/coco/val2017
export COCO_VAL_ANN_JSON=/path/to/coco/annotations/instances_val2017.json
```

Interactive:
```bash
bash scripts/run_coco_finetune_blip2.sh
```

Slurm:
```bash
sbatch --account=def-XXXX --export=ALL,COCO_TRAIN_IMAGES_DIR=...,COCO_TRAIN_ANN_JSON=...,COCO_VAL_IMAGES_DIR=...,COCO_VAL_ANN_JSON=... slurm/coco_finetune_blip2.sbatch
```

Hyperparameter overrides (offline W&B logs them):
```bash
sbatch --account=def-XXXX --export=ALL,COCO_TRAIN_IMAGES_DIR=...,COCO_TRAIN_ANN_JSON=...,COCO_VAL_IMAGES_DIR=...,COCO_VAL_ANN_JSON=...,LR=1e-4,BS=32,WD=0.05,OPT=adamw,EPOCHS=10 slurm/coco_finetune_blip2.sbatch
```

## 4) Medical transfer (mammography / neuroimaging via CSV)
CSV format:
- `image_path` column
- binary: `label` in {0,1}
- multi-label: `labels` as comma-separated indices (e.g., `0,3,7`)

Run:
```bash
sbatch --account=def-XXXX --export=ALL,CFG=$HOME/projects/blip2_interleaved_project/configs/medical_finetune_mammo.yaml slurm/medical_finetune.sbatch
```

## Outputs
Runs are stored under:
- `${SCRATCH}/blip2_interleaved_project/runs/`
Each run contains:
- `config_resolved.yaml`
- `metrics.jsonl`
- `checkpoints/best.ckpt` (best validation score)
- `wandb/` (offline W&B logs)

## Notes
- If Narval outbound internet is restricted, pre-download checkpoints to `${HF_HOME}` / `${TORCH_HOME}` and point configs/env vars accordingly.
- For BioViL / Mammo-CLIP, set the relevant env variables (see scripts).
