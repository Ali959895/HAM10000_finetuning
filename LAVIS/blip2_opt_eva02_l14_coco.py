#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
blip2_opt_eva02_l14_coco.py
Offline-first BLIP-2 (OPT-2.7B) training on COCO captions with:
- EVA02-L/14 vision encoder (via LAVIS config/model)
- LoRA on OPT only (PEFT)
- W&B offline logging

IMPORTANT (offline):
- You MUST have all model weights locally (no HF downloads).
- This script sets TRANSFORMERS_OFFLINE=1 and HF_HUB_OFFLINE=1.
- It will fail fast with clear errors if paths are missing.

Example:
python blip2_opt_eva02_l14_coco.py \
  --lavis_repo /home/ali95/LAVIS/LAVIS-main \
  --coco_root /scratch/ali95/coco \
  --ann_train /scratch/ali95/coco/annotations/captions_train2017.json \
  --ann_val /scratch/ali95/coco/annotations/captions_val2017.json \
  --images_train /scratch/ali95/coco/train2017 \
  --images_val /scratch/ali95/coco/val2017 \
  --output_dir /scratch/ali95/blip2_runs/blip2_opt2.7b_eva02l14_coco \
  --batch_size 8 \
  --accum_steps 4 \
  --epochs 2 \
  --bf16 \
  --use_wandb \
  --wandb_offline \
  --wandb_project blip2_coco \
  --wandb_run_name blip2_opt27b_eva02l14_lora
"""

import os
import sys
import json
import math
import time
import argparse
from dataclasses import dataclass
from typing import Dict, Any, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from PIL import Image
# ===== OFFLINE EVA02-L/14 FIX =====
os.environ["TIMM_HUB_DIR"] = "/scratch/ali95/models"
os.environ["HF_HOME"] = "/scratch/ali95/hf_cache"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
# =================================

# -------------------------
# Offline hardening
# -------------------------
def force_offline_env(hf_home: str):
    os.environ.setdefault("HF_HOME", hf_home)
    os.environ.setdefault("TRANSFORMERS_CACHE", os.path.join(hf_home, "transformers"))  # ok if set; HF_HOME preferred
    os.environ.setdefault("HF_HUB_CACHE", os.path.join(hf_home, "hub"))
    os.environ.setdefault("HF_DATASETS_CACHE", os.path.join(hf_home, "datasets"))

    # Hard offline:
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["HF_HUB_OFFLINE"] = "1"

    # W&B behavior handled separately via flags/env in main

# -------------------------
# Utilities
# -------------------------
def log(msg: str):
    print(msg, flush=True)

def set_seed(seed: int = 42):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_rank0() -> bool:
    return (not torch.distributed.is_available()) or (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0

def maybe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)

# -------------------------
# COCO captions dataset
# -------------------------
class CocoCaptionsDataset(Dataset):
    """
    Minimal COCO captions dataset (no external deps).
    Reads captions_*.json and yields (PIL image, caption str).
    """

    def __init__(self, images_dir: str, ann_json: str, vis_processor):
        self.images_dir = images_dir
        self.ann_json = ann_json
        self.vis_processor = vis_processor

        if not os.path.isfile(ann_json):
            raise FileNotFoundError(f"Annotation JSON not found: {ann_json}")
        if not os.path.isdir(images_dir):
            raise FileNotFoundError(f"Images directory not found: {images_dir}")

        with open(ann_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        # image_id -> file_name
        id2file = {img["id"]: img["file_name"] for img in data["images"]}

        # build (img_path, caption)
        samples = []
        for ann in data["annotations"]:
            img_id = ann["image_id"]
            cap = ann["caption"]
            fn = id2file.get(img_id, None)
            if fn is None:
                continue
            path = os.path.join(images_dir, fn)
            samples.append((path, cap))

        if len(samples) == 0:
            raise RuntimeError(f"No samples found in {ann_json} with images_dir={images_dir}")

        self.samples = samples
        log(f"[data] Loaded {len(self.samples)} caption pairs from {ann_json}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, caption = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.vis_processor(img)
        return img, caption

@dataclass
class Batch:
    images: torch.Tensor
    captions: List[str]

def collate_fn(batch) -> Batch:
    imgs, caps = zip(*batch)
    return Batch(images=torch.stack(list(imgs), dim=0), captions=list(caps))

# -------------------------
# LAVIS loading
# -------------------------
def add_lavis_to_path(lavis_repo: str):
    """
    lavis_repo should point to LAVIS-main (contains lavis/).
    """
    if not os.path.isdir(lavis_repo):
        raise FileNotFoundError(f"--lavis_repo not found: {lavis_repo}")
    if not os.path.isdir(os.path.join(lavis_repo, "lavis")):
        raise FileNotFoundError(f"--lavis_repo must contain lavis/ package: {lavis_repo}")
    sys.path.insert(0, lavis_repo)

def load_blip2_opt_from_lavis(device: torch.device, model_type: str):
    """
    Uses LAVIS official API.
    model_type examples (depends on your LAVIS version):
      - "pretrain_opt2.7b"
      - "caption_coco_opt2.7b"
    """
    from lavis.models import load_model_and_preprocess

    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_opt",
        model_type=model_type,
        is_eval=False,
        device=device,
    )
    return model, vis_processors, txt_processors

# -------------------------
# LoRA (OPT only)
# -------------------------
def apply_lora_to_opt_only(
    blip2_model,
    r=8,
    alpha=16,
    dropout=0.05,
    target_modules=None,
):
    from peft import LoraConfig, get_peft_model

    if target_modules is None:
        target_modules = ["q_proj", "k_proj", "v_proj", "out_proj"]

    # 1️⃣ Freeze everything
    for p in blip2_model.parameters():
        p.requires_grad = False

    # 2️⃣ LoRA config
    lora_config = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=target_modules,
        task_type="CAUSAL_LM",
    )

    # 3️⃣ Apply LoRA ONLY to OPT
    blip2_model.opt_model = get_peft_model(
        blip2_model.opt_model,
        lora_config
    )

    # 4️⃣ Stats
    trainable = sum(p.numel() for p in blip2_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in blip2_model.parameters())

    print("✅ LoRA applied to OPT only")
    print(f"Trainable params: {trainable:,}")
    print(f"Total params:     {total:,}")
    print(f"Trainable %:      {100 * trainable / total:.4f}%")

    return blip2_model, trainable, total


# -------------------------
# Training / eval
# -------------------------
@torch.no_grad()
def run_val(model: nn.Module, loader: DataLoader, device: torch.device, bf16: bool) -> float:
    model.eval()
    losses = []
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if (bf16 and device.type == "cuda") else nullcontext()
    for batch in loader:
        images = batch.images.to(device, non_blocking=True)
        # LAVIS BLIP2 models typically accept dict inputs, but BLIP2-OPT in LAVIS usually expects:
        # samples = {"image": images, "text_input": captions}
        samples = {"image": images, "text_input": batch.captions}
        with autocast_ctx:
            out = model(samples)
            # LAVIS returns dict with "loss" often
            loss = out["loss"] if isinstance(out, dict) and "loss" in out else out
        losses.append(loss.detach().float().item())
    model.train()
    return float(sum(losses) / max(1, len(losses)))

def save_checkpoint(model: nn.Module, optimizer: torch.optim.Optimizer, step: int, outdir: str):
    maybe_mkdir(outdir)
    path = os.path.join(outdir, f"checkpoint_step{step}.pt")
    # Save only trainable + state for reproducibility
    torch.save(
        {
            "step": step,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        },
        path,
    )
    return path

# contextlib.nullcontext for older pythons
try:
    from contextlib import nullcontext
except Exception:
    class nullcontext:
        def __enter__(self): return None
        def __exit__(self, *args): return False

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lavis_repo", type=str, required=True, help="Path to LAVIS-main (contains lavis/)")
    ap.add_argument("--coco_root", type=str, required=False, default=None, help="Optional, for your bookkeeping")
    ap.add_argument("--ann_train", type=str, required=True)
    ap.add_argument("--ann_val", type=str, required=True)
    ap.add_argument("--images_train", type=str, required=True)
    ap.add_argument("--images_val", type=str, required=True)

    ap.add_argument("--output_dir", type=str, required=True)
    ap.add_argument("--hf_home", type=str, default="/scratch/ali95/hf_home")

    ap.add_argument("--model_type", type=str, default="pretrain_opt2.7b",
                    help="LAVIS model_type for blip2_opt (depends on your LAVIS version)")

    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--warmup_steps", type=int, default=200)
    ap.add_argument("--max_steps", type=int, default=0, help="0 means use full epochs")
    ap.add_argument("--accum_steps", type=int, default=1)
    ap.add_argument("--grad_clip", type=float, default=1.0)

    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--seed", type=int, default=42)

    # LoRA
    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", type=str, default="q_proj,v_proj",
                    help="Comma-separated target module names inside OPT attention (often q_proj,k_proj,v_proj,out_proj)")

    # Logging / checkpoint
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--val_every", type=int, default=500)
    ap.add_argument("--ckpt_every", type=int, default=1000)

    # W&B
    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--wandb_offline", action="store_true")
    ap.add_argument("--wandb_project", type=str, default="blip2_coco")
    ap.add_argument("--wandb_run_name", type=str, default="blip2_opt2.7b_eva02l14_lora")
    ap.add_argument("--wandb_dir", type=str, default=None)

    args = ap.parse_args()

    # Offline env
    force_offline_env(args.hf_home)

    # W&B offline mode
    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"
    # (Optional) keep wandb files on scratch
    if args.wandb_dir:
        os.environ["WANDB_DIR"] = args.wandb_dir

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[device] {device}")

    maybe_mkdir(args.output_dir)

    # Add LAVIS
    add_lavis_to_path(args.lavis_repo)

    # Load model + preprocessors
    log(f"[model] Loading LAVIS blip2_opt model_type={args.model_type} (offline expected)")
    model, vis_processors, _ = load_blip2_opt_from_lavis(device=device, model_type=args.model_type)

    # Use the "train" visual processor if available, else fallback to "eval"
    vis_proc = vis_processors.get("train", None) or vis_processors.get("eval", None)
    if vis_proc is None:
        raise RuntimeError("Could not find vis_processors['train'] or ['eval'] from LAVIS.")

    # Apply LoRA only on OPT LM
    if args.use_lora:
        targets = [t.strip() for t in args.lora_targets.split(",") if t.strip()]
        log(f"[lora] Applying LoRA to OPT only. targets={targets}, r={args.lora_r}, alpha={args.lora_alpha}, drop={args.lora_dropout}")
        model, trainable, total = apply_lora_to_opt_only(
            model,
            r=args.lora_r,
            alpha=args.lora_alpha,
            dropout=args.lora_dropout,
            target_modules=targets,
        )
        log(f"[lora] Trainable params: {trainable/1e6:.2f}M / Total: {total/1e6:.2f}M")
    else:
        # If not LoRA, you probably still want to freeze most and tune some head;
        # but user requested LoRA-on-OPT only, so keep default as-is (LAVIS may already freeze some parts).
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        log(f"[train] Trainable params (no LoRA): {trainable/1e6:.2f}M / Total: {total/1e6:.2f}M")

    model.train()

    # Optimizer on trainable params only
    params = [p for p in model.parameters() if p.requires_grad]
    if len(params) == 0:
        raise RuntimeError("No trainable parameters found. If using LoRA, ensure peft applied correctly.")

    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)

    # Datasets
    train_ds = CocoCaptionsDataset(args.images_train, args.ann_train, vis_proc)
    val_ds = CocoCaptionsDataset(args.images_val, args.ann_val, vis_proc)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=max(1, args.num_workers // 2),
        pin_memory=(device.type == "cuda"),
        collate_fn=collate_fn,
        drop_last=False,
    )

    # W&B init (offline)
    wandb = None
    if args.use_wandb and is_rank0():
        try:
            import wandb as _wandb
            wandb = _wandb
            wandb.init(
                project=args.wandb_project,
                name=args.wandb_run_name,
                config=vars(args),
            )
            log("[wandb] initialized (offline if WANDB_MODE=offline)")
        except Exception as e:
            log(f"[wandb] disabled due to import/init error: {e}")
            wandb = None

    # Training loop
    step = 0
    t0 = time.time()

    # Steps per epoch
    steps_per_epoch = len(train_loader)
    if args.max_steps and args.max_steps > 0:
        total_steps = args.max_steps
        total_epochs = math.ceil(total_steps / max(1, steps_per_epoch))
    else:
        total_epochs = args.epochs
        total_steps = total_epochs * steps_per_epoch

    log(f"[train] epochs={total_epochs}, steps_per_epoch={steps_per_epoch}, total_steps={total_steps}")
    autocast_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if (args.bf16 and device.type == "cuda") else nullcontext()

    optimizer.zero_grad(set_to_none=True)

    for epoch in range(total_epochs):
        for batch in train_loader:
            step += 1
            images = batch.images.to(device, non_blocking=True)
            samples = {"image": images, "text_input": batch.captions}

            with autocast_ctx:
                out = model(samples)
                loss = out["loss"] if isinstance(out, dict) and "loss" in out else out

            # normalize by accumulation
            loss = loss / args.accum_steps
            loss.backward()

            if step % args.accum_steps == 0:
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

            # logging
            if (step % args.log_every == 0) and is_rank0():
                lr = optimizer.param_groups[0]["lr"]
                loss_val = float(loss.detach().float().item() * args.accum_steps)
                elapsed = time.time() - t0
                it_s = elapsed / max(1, step)
                msg = f"[step {step}/{total_steps}] loss={loss_val:.4f} lr={lr:.2e} sec/it={it_s:.3f}"
                log(msg)
                if wandb is not None:
                    wandb.log(
                        {
                            "train/loss": loss_val,
                            "train/lr": lr,
                            "sys/sec_per_iter": it_s,
                        },
                        step=step,
                    )

            # validation
            if (args.val_every > 0) and (step % args.val_every == 0) and is_rank0():
                val_loss = run_val(model, val_loader, device, bf16=args.bf16)
                log(f"[val] step={step} val_loss={val_loss:.4f}")
                if wandb is not None:
                    wandb.log({"val/loss": val_loss}, step=step)

            # checkpoint
            if (args.ckpt_every > 0) and (step % args.ckpt_every == 0) and is_rank0():
                ckpt_path = save_checkpoint(model, optimizer, step, args.output_dir)
                log(f"[ckpt] saved: {ckpt_path}")
                if wandb is not None:
                    wandb.log({"sys/ckpt_saved": 1}, step=step)

            if args.max_steps and step >= args.max_steps:
                break

        if args.max_steps and step >= args.max_steps:
            break

    # Final save
    if is_rank0():
        ckpt_path = save_checkpoint(model, optimizer, step, args.output_dir)
        log(f"[final] saved: {ckpt_path}")

    if wandb is not None and is_rank0():
        wandb.finish()

    log("[done]")

if __name__ == "__main__":
    main()
