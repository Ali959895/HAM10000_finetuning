#!/bin/bash
set -euo pipefail

COUNT="${1:-1}"

# ✅ PRE-CREATED SWEEP ID (DO NOT CHANGE)
SWEEP_ID="ali-algumaei-concordia-university/uncategorized/s0n0ywns"

echo "=== W&B AGENT LAUNCH ==="
echo "Sweep ID : ${SWEEP_ID}"
echo "Runs     : ${COUNT}"
echo "PWD      : $(pwd)"
echo "======================="

# 🚨 IMPORTANT:
# - NO python command here
# - NO train_coco.py
# - NO model_name / model_type
# All of that lives INSIDE the sweep YAML

wandb agent "${SWEEP_ID}" --count "${COUNT}"
