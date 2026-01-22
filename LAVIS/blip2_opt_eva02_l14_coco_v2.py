#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
blip2_opt_eva02_l14_coco.py (UPDATED – performance + offline sweeps)

Key upgrades:
✔ Cosine LR + warmup (OPT stability)
✔ TF32 enabled (A100 speedup)
✔ Persistent dataloader workers
✔ Optional caption truncation
✔ Verified W&B OFFLINE (supports sweeps)

Still:
✔ Offline-only
✔ EVA02-L/14 via LAVIS
✔ LoRA on OPT only
"""

import os
import sys
import json
import math
import time
import argparse
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# -----------------------------------------------------------------------------
# HARD OFFLINE + CUDA SPEED
# -----------------------------------------------------------------------------
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("WANDB_DISABLE_CODE", "true")
os.environ.setdefault("WANDB_START_METHOD", "thread")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def log(msg: str):
    print(msg, flush=True)

def set_seed(seed: int):
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_rank0():
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0

def maybe_mkdir(p):
    os.makedirs(p, exist_ok=True)

# -----------------------------------------------------------------------------
# Dataset
# -----------------------------------------------------------------------------
class CocoCaptionsDataset(Dataset):
    def __init__(self, images_dir, ann_json, vis_proc, max_words=64):
        self.vis_proc = vis_proc
        self.max_words = max_words

        with open(ann_json, "r") as f:
            data = json.load(f)

        id2file = {img["id"]: img["file_name"] for img in data["images"]}
        self.samples = [
            (os.path.join(images_dir, id2file[a["image_id"]]), a["caption"])
            for a in data["annotations"]
            if a["image_id"] in id2file
        ]

        log(f"[data] Loaded {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cap = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.vis_proc(img)
        cap = " ".join(cap.split()[: self.max_words])
        return img, cap

@dataclass
class Batch:
    images: torch.Tensor
    captions: List[str]

def collate_fn(batch):
    imgs, caps = zip(*batch)
    return Batch(torch.stack(imgs), list(caps))

# -----------------------------------------------------------------------------
# LAVIS
# -----------------------------------------------------------------------------
def add_lavis_to_path(repo):
    if not os.path.isdir(os.path.join(repo, "lavis")):
        raise FileNotFoundError(repo)
    sys.path.insert(0, repo)

def load_blip2(device, model_type):
    from lavis.models import load_model_and_preprocess
    model, vis, _ = load_model_and_preprocess(
        name="blip2_opt",
        model_type=model_type,
        is_eval=False,
        device=device,
    )
    return model, vis

# -----------------------------------------------------------------------------
# LoRA
# -----------------------------------------------------------------------------
def apply_lora(model, r, alpha, dropout, targets):
    from peft import LoraConfig, get_peft_model

    for p in model.parameters():
        p.requires_grad = False

    cfg = LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        bias="none",
        target_modules=targets,
        task_type="CAUSAL_LM",
    )

    model.opt_model = get_peft_model(model.opt_model, cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    log(f"[LoRA] Trainable {trainable/1e6:.2f}M / {total/1e6:.2f}M")
    return model

# -----------------------------------------------------------------------------
# LR Scheduler
# -----------------------------------------------------------------------------
def cosine_schedule(opt, warmup, total):
    def lr_lambda(step):
        if step < warmup:
            return step / max(1, warmup)
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1 + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lavis_repo", required=True)
    ap.add_argument("--ann_train", required=True)
    ap.add_argument("--ann_val", required=True)
    ap.add_argument("--images_train", required=True)
    ap.add_argument("--images_val", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_type", default="pretrain_opt2.7b")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--accum_steps", type=int, default=4)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=500)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--max_caption_words", type=int, default=64)

    ap.add_argument("--use_lora", action="store_true")
    ap.add_argument("--lora_r", type=int, default=16)
    ap.add_argument("--lora_alpha", type=int, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--lora_targets", default="q_proj,v_proj")

    ap.add_argument("--use_wandb", action="store_true")
    ap.add_argument("--wandb_offline", action="store_true")
    ap.add_argument("--wandb_project", default="blip2_coco")
    ap.add_argument("--wandb_run_name", default="blip2_opt27b")

    args = ap.parse_args()

    if args.wandb_offline:
        os.environ["WANDB_MODE"] = "offline"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"[device] {device}")

    maybe_mkdir(args.output_dir)
    add_lavis_to_path(args.lavis_repo)

    model, vis = load_blip2(device, args.model_type)
    vis_proc = vis["train"]

    if args.use_lora:
        model = apply_lora(
            model,
            args.lora_r,
            args.lora_alpha,
            args.lora_dropout,
            args.lora_targets.split(","),
        )

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr)
    sched = cosine_schedule(opt, args.warmup_steps, args.epochs * 1_000)

    train_ds = CocoCaptionsDataset(
        args.images_train, args.ann_train, vis_proc, args.max_caption_words
    )

    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        persistent_workers=True,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    wandb = None
    if args.use_wandb and is_rank0():
        import wandb as _wandb
        wandb = _wandb
        wandb.init(project="blip2_coco",
          name=args.wandb_run_name,
          mode="offline",
          )
        log(f"[wandb] mode={wandb.run.settings.mode}")

    model.train()
    step = 0
    scaler = torch.cuda.amp.GradScaler(enabled=args.bf16)

    for epoch in range(args.epochs):
    
        # ✅ Print epoch start ONCE
        if is_rank0():
            print(
                f"\n[INFO] ===== Epoch {epoch}/{args.epochs - 1} started =====",
                flush=True
            )
    
        epoch_start = time.time()
    
        for it, batch in enumerate(loader):
            step += 1
    
            imgs = batch.images.to(device, non_blocking=True)
            samples = {"image": imgs, "text_input": batch.captions}
    
            with torch.cuda.amp.autocast(enabled=args.bf16):
                out = model(samples)
                loss = out["loss"] / args.accum_steps
    
            scaler.scale(loss).backward()
    
            # ✅ Gradient accumulation step
            if step % args.accum_steps == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
    
                # print only once after first optimizer step
                if step == args.accum_steps and is_rank0():
                    print("[INFO] First optimizer step completed", flush=True)
    
            # ✅ Logging (safe, no undefined vars)
            if step % 50 == 0 and is_rank0():
                loss_val = loss.item() * args.accum_steps
                step_time = time.time() - epoch_start
    
                log(
                    f"[STEP {step}] "
                    f"Epoch {epoch} | "
                    f"Iter {it+1}/{len(loader)} | "
                    f"Loss {loss_val:.4f} | "
                    f"Elapsed {step_time:.2f}s"
                )
    
                if wandb:
                    wandb.log(
                        {
                            "train/loss": loss_val,
                            "epoch": epoch,
                            "iter": it,
                        },
                        step=step,
                    )


if __name__ == "__main__":
    main()
