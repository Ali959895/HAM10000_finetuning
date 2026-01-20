#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
BLIP-2 OPT-2.7B COCO Training (Compute Canada Safe)
- W&B OFFLINE ONLY
- LoRA on OPT only
- Reduced Vision Encoder Depth
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import wandb

# ---------------------------------------------------------------------
# Environment safety (NO INTERNET)
# ---------------------------------------------------------------------
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# ---------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser("BLIP2 OPT-2.7B COCO")

    # Data / paths
    parser.add_argument("--coco_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    # Training
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--accum_steps", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num_workers", type=int, default=4)

    # Precision
    parser.add_argument("--bf16", action="store_true")

    # OPTION A — speed
    parser.add_argument("--reduce_vision_blocks", type=int, default=None)

    # OPTION C — LoRA
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)

    # Logging
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--val_every", type=int, default=2000)
    parser.add_argument("--ckpt_every", type=int, default=5000)

    return parser.parse_args()


# ---------------------------------------------------------------------
# W&B initialization (OFFLINE ONLY)
# ---------------------------------------------------------------------
def init_wandb(args):
    wandb.init(
        project="blip2_opt_coco",
        entity="ali-algumaei-concordia-university",
        name=f"opt27b_lora_r{args.lora_r}_lr{args.lr}",
        mode="offline",
        config=vars(args),
        dir=os.environ.get("WANDB_DIR", "./wandb"),
    )
    print("[wandb] initialized (offline)")


# ---------------------------------------------------------------------
# Load BLIP2 from LAVIS
# ---------------------------------------------------------------------
def load_blip2(device):
    from lavis.models import load_model_and_preprocess

    print("[model] Loading LAVIS blip2_opt model_type=pretrain_opt2.7b (offline expected)")

    model, vis_processors, txt_processors = load_model_and_preprocess(
        name="blip2_opt",
        model_type="pretrain_opt2.7b",
        is_eval=False,
        device=device,
    )
    return model, vis_processors, txt_processors


# ---------------------------------------------------------------------
# OPTION A — Reduce Vision Encoder Depth
# ---------------------------------------------------------------------
def reduce_vision_blocks(model, n_blocks):
    if n_blocks is None:
        return

    ve = model.visual_encoder
    if hasattr(ve, "blocks"):
        ve.blocks = ve.blocks[:n_blocks]
        print(f"[INFO] Vision encoder blocks reduced to {n_blocks}")
    else:
        print("[WARN] visual_encoder has no blocks attribute")


# ---------------------------------------------------------------------
# OPTION C — Apply LoRA to OPT only
# ---------------------------------------------------------------------
def apply_lora(model, args):
    from peft import LoraConfig, get_peft_model

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model.opt_model = get_peft_model(model.opt_model, lora_cfg)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    print("? LoRA applied to OPT only")
    print(f"Trainable params: {trainable:,}")
    print(f"Total params:     {total:,}")
    print(f"Trainable %:      {100 * trainable / total:.4f}%")

    return model


# ---------------------------------------------------------------------
# Dummy COCO loader hook (replace with your existing one)
# ---------------------------------------------------------------------
def build_dataloaders(coco_root, batch_size, num_workers):
    from lavis.datasets.builders import load_dataset_config
    from lavis.datasets.builders.coco_caption_builder import COCOCapBuilder

    builder = COCOCapBuilder()
    datasets = builder.build_datasets()

    train_loader = DataLoader(
        datasets["train"],
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        datasets["val"],
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader


# ---------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------
@torch.no_grad()
def run_validation(model, val_loader, device):
    model.eval()
    losses = []

    for samples in val_loader:
        samples = {k: v.to(device) if torch.is_tensor(v) else v for k, v in samples.items()}
        out = model(samples)
        losses.append(out["loss"].item())

    return sum(losses) / len(losses)


# ---------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    print(f"[device] {device}")

    # Init logging
    init_wandb(args)

    # Load model
    model, vis_processors, txt_processors = load_blip2(device)

    # OPTION A
    reduce_vision_blocks(model, args.reduce_vision_blocks)

    # OPTION C
    if args.use_lora:
        model = apply_lora(model, args)

    model.to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=args.bf16)

    # Data
    train_loader, val_loader = build_dataloaders(
        args.coco_root, args.batch_size, args.num_workers
    )

    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    print(f"[train] epochs={args.epochs}, steps_per_epoch={steps_per_epoch}, total_steps={total_steps}")

    global_step = 0
    start_time = time.time()

    # Training
    for epoch in range(args.epochs):
        model.train()

        for samples in train_loader:
            samples = {k: v.to(device) if torch.is_tensor(v) else v for k, v in samples.items()}

            with torch.cuda.amp.autocast(enabled=args.bf16):
                out = model(samples)
                loss = out["loss"] / args.accum_steps

            scaler.scale(loss).backward()

            if (global_step + 1) % args.accum_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            if global_step % args.log_every == 0:
                sec_per_it = (time.time() - start_time) / max(1, global_step + 1)
                print(
                    f"[step {global_step}/{total_steps}] "
                    f"loss={loss.item() * args.accum_steps:.4f} "
                    f"lr={args.lr:.2e} sec/it={sec_per_it:.3f}"
                )
                wandb.log(
                    {
                        "train/loss": loss.item() * args.accum_steps,
                        "lr": args.lr,
                        "sec_per_it": sec_per_it,
                        "step": global_step,
                    }
                )

            if global_step > 0 and global_step % args.val_every == 0:
                val_loss = run_validation(model, val_loader, device)
                print(f"[val] step={global_step} val_loss={val_loss:.4f}")
                wandb.log({"val/loss": val_loss, "step": global_step})

            if global_step > 0 and global_step % args.ckpt_every == 0:
                ckpt = Path(args.output_dir) / f"checkpoint_step{global_step}.pt"
                torch.save({"model": model.state_dict()}, ckpt)
                print(f"[ckpt] saved: {ckpt}")

            global_step += 1

    # Final save
    final_ckpt = Path(args.output_dir) / f"checkpoint_step{global_step}.pt"
    torch.save({"model": model.state_dict()}, final_ckpt)
    print(f"[final] saved: {final_ckpt}")

    wandb.finish()
    print("[done]")


if __name__ == "__main__":
    main()
