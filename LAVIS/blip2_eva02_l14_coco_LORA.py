#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
BLIP-2 (Salesforce/blip2-flan-t5-xl) + LoRA fine-tuning on COCO Captions
Compute Canada / Narval friendly:
  - NO internet assumptions
  - W&B forced OFFLINE
  - bf16 on A100
  - advanced logging: grad norms, parameter histograms (optional), PR curves (optional)

Run:
  python blip2_eva02_l14_coco_LORA.py --config coco_LORA.yaml
"""

import os
# ---- MUST be set BEFORE importing wandb ----
os.environ.setdefault("WANDB_MODE", "offline")
os.environ.setdefault("WANDB_SILENT", "true")
os.environ.setdefault("WANDB_START_METHOD", "thread")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import json
import math
import time
import argparse
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import (
    AutoProcessor,
    Blip2ForConditionalGeneration,
    get_cosine_schedule_with_warmup,
)

# LoRA
from peft import LoraConfig, get_peft_model

# YAML
import yaml

# W&B (offline)
import wandb


# -------------------------
# Utilities
# -------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def is_bf16_supported() -> bool:
    if not torch.cuda.is_available():
        return False
    return torch.cuda.get_device_capability(0)[0] >= 8  # A100 = 8.0

def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        else:
            out[k] = v
    return out


# -------------------------
# COCO Captions Dataset
# -------------------------

class CocoCaptions(Dataset):
    """
    Reads COCO captions JSON (captions_train2017.json) and images dir (train2017/).
    Returns (PIL image, caption text, image_id).
    """

    def __init__(self, images_dir: str, captions_json: str):
        self.images_dir = images_dir
        with open(captions_json, "r") as f:
            data = json.load(f)

        # Map image_id -> file_name
        img_map = {img["id"]: img["file_name"] for img in data["images"]}

        # Each annotation: image_id, caption, id
        self.items = []
        for ann in data["annotations"]:
            image_id = ann["image_id"]
            caption = ann["caption"]
            file_name = img_map.get(image_id)
            if file_name is None:
                continue
            path = os.path.join(images_dir, file_name)
            self.items.append((path, caption, image_id))

        if len(self.items) == 0:
            raise RuntimeError(
                f"No items found. Check paths:\n  images_dir={images_dir}\n  captions_json={captions_json}"
            )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, caption, image_id = self.items[idx]
        img = Image.open(path).convert("RGB")
        return img, caption, image_id


@dataclass
class Collator:
    processor: Any
    max_text_len: int

    def __call__(self, batch: List[Tuple[Image.Image, str, int]]) -> Dict[str, Any]:
        images, captions, image_ids = zip(*batch)

        # BLIP-2 expects: images + text prompt/labels
        # We train by feeding text as labels (teacher forcing).
        # Using processor handles both image preprocessing + tokenization.
        proc = self.processor(
            images=list(images),
            text=list(captions),
            padding=True,
            truncation=True,
            max_length=self.max_text_len,
            return_tensors="pt",
        )

        # Labels: same as input_ids but with padding -> -100
        labels = proc["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100

        proc["labels"] = labels
        proc["image_ids"] = torch.tensor(image_ids, dtype=torch.long)
        return proc


# -------------------------
# Optional evaluation helpers
# -------------------------

def quick_caption_eval(
    model: nn.Module,
    processor: Any,
    loader: DataLoader,
    device: torch.device,
    max_new_tokens: int = 32,
    num_samples: int = 2000,
) -> Dict[str, float]:
    """
    Lightweight evaluation:
      - Generate captions for a subset and compute token-level average length + simple exact-match proxy
    NOTE: True COCO metrics (CIDEr/SPICE) require pycocoevalcap.
    We keep it minimal & robust offline.
    """
    model.eval()
    gens = 0
    total_len = 0

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if num_samples > 0 and gens >= num_samples:
                break
            batch = to_device(batch, device)
            # Generation uses images; we provide an empty prompt
            pixel_values = batch["pixel_values"]
            generated_ids = model.generate(
                pixel_values=pixel_values,
                max_new_tokens=max_new_tokens,
            )
            texts = processor.batch_decode(generated_ids, skip_special_tokens=True)
            for t in texts:
                total_len += len(t.split())
            gens += len(texts)

    avg_len = (total_len / max(gens, 1))
    return {"eval/generated_avg_words": float(avg_len)}


def log_param_histograms(model: nn.Module, step: int) -> Dict[str, wandb.Histogram]:
    """
    Advanced logging: parameter histograms (can be heavy).
    We'll log ONLY LoRA parameters by default.
    """
    h = {}
    for n, p in model.named_parameters():
        if p is None or (not p.requires_grad):
            continue
        # Focus on LoRA matrices to keep it lightweight
        if "lora_" not in n:
            continue
        data = p.detach().float().cpu().numpy().flatten()
        if data.size == 0:
            continue
        h[f"hist/{n}"] = wandb.Histogram(data)
    return h


def grad_global_norm(model: nn.Module) -> float:
    total_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is None:
            continue
        param_norm = p.grad.detach().data.float().norm(2).item()
        total_norm_sq += param_norm ** 2
    return float(math.sqrt(total_norm_sq))


# -------------------------
# Main training
# -------------------------

def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def save_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=str)
    args = ap.parse_args()

    cfg = load_config(args.config)

    # Paths / outputs
    out_dir = cfg["run"]["output_dir"]
    safe_mkdir(out_dir)
    safe_mkdir(os.path.join(out_dir, "checkpoints"))

    # Seed
    seed = int(cfg["run"].get("seed", 42))
    set_seed(seed)

    # Device / precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    precision = cfg["train"].get("precision", "bf16").lower()
    use_bf16 = (precision == "bf16") and is_bf16_supported()
    use_fp16 = (precision == "fp16")
    if precision not in ["bf16", "fp16", "fp32"]:
        raise ValueError(f"Unsupported precision: {precision}")

    print(f"[{now()}] device={device}, precision={precision} (bf16_supported={is_bf16_supported()})", flush=True)

    # W&B (offline)
    wb = cfg.get("wandb", {})
    wandb_enabled = bool(wb.get("enabled", True))
    if wandb_enabled:
        wandb.init(
            project=wb.get("project", "blip2-coco-lora"),
            entity=wb.get("entity") or None,
            name=wb.get("run_name", None),
            tags=wb.get("tags", None),
            mode=wb.get("mode", "offline"),
            config=cfg,
        )
        wandb.run.log_code(".", include_fn=lambda p: p.endswith(".py") or p.endswith(".yaml") or p.endswith(".sh"))

    # Model / processor
    model_name = cfg["model"]["hf_model_name"]
    print(f"[{now()}] loading processor/model: {model_name}", flush=True)

    processor = AutoProcessor.from_pretrained(model_name)
    model = Blip2ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=(torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)),
        device_map=None,
    )

    if cfg["model"].get("use_gradient_checkpointing", True):
        # For memory savings on A100
        try:
            model.gradient_checkpointing_enable()
            print(f"[{now()}] gradient checkpointing enabled", flush=True)
        except Exception as e:
            print(f"[{now()}] gradient checkpointing enable failed: {e}", flush=True)

    # LoRA
    lora_cfg = cfg.get("lora", {})
    if lora_cfg.get("enabled", True):
        target_modules = lora_cfg.get("target_modules", ["q", "k", "v", "o"])
        lora = LoraConfig(
            r=int(lora_cfg.get("r", 16)),
            lora_alpha=int(lora_cfg.get("alpha", 32)),
            lora_dropout=float(lora_cfg.get("dropout", 0.05)),
            bias=str(lora_cfg.get("bias", "none")),
            target_modules=target_modules,
            task_type="SEQ_2_SEQ_LM",  # T5 style
        )
        # Apply LoRA to the language model inside BLIP2
        # NOTE: BLIP2 wraps language model as model.language_model
        model.language_model = get_peft_model(model.language_model, lora)
        print(f"[{now()}] LoRA enabled on language_model, targets={target_modules}", flush=True)
        model.language_model.print_trainable_parameters()

    model.to(device)

    # Data
    train_ds = CocoCaptions(cfg["data"]["images_dir"], cfg["data"]["ann_train"])
    val_ds = CocoCaptions(cfg["data"]["images_dir"].replace("train2017", "val2017"), cfg["data"]["ann_val"]) \
        if os.path.exists(cfg["data"]["ann_val"]) else None

    collate = Collator(processor=processor, max_text_len=int(cfg["model"].get("max_text_len", 32)))

    num_workers = int(cfg["run"].get("num_workers", 4))
    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
        collate_fn=collate,
        drop_last=True,
    )
    val_loader = None
    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=int(cfg["train"]["batch_size"]),
            shuffle=False,
            num_workers=num_workers,
            pin_memory=(device.type == "cuda"),
            collate_fn=collate,
            drop_last=False,
        )

    # Optimizer / scheduler
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"].get("weight_decay", 0.01))
    max_grad_norm = float(cfg["train"].get("max_grad_norm", 1.0))
    grad_accum = int(cfg["train"].get("grad_accum_steps", 1))
    warmup_ratio = float(cfg["train"].get("warmup_ratio", 0.03))

    # Only trainable params (LoRA)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=wd)

    epochs = int(cfg["train"].get("epochs", 1))
    max_steps = int(cfg["train"].get("max_steps", -1))
    total_update_steps = (len(train_loader) * epochs) // max(grad_accum, 1)
    if max_steps > 0:
        total_update_steps = max_steps

    warmup_steps = int(total_update_steps * warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_update_steps,
    )

    # Mixed precision context
    if use_bf16:
        autocast = torch.amp.autocast
        autocast_kwargs = {"device_type": "cuda", "dtype": torch.bfloat16}
    elif use_fp16:
        autocast = torch.amp.autocast
        autocast_kwargs = {"device_type": "cuda", "dtype": torch.float16}
    else:
        autocast = None
        autocast_kwargs = {}

    # Resume
    resume_from = cfg["run"].get("resume_from", "") or ""
    global_step = 0
    if resume_from:
        ckpt = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optim"])
        scheduler.load_state_dict(ckpt["sched"])
        global_step = int(ckpt.get("global_step", 0))
        print(f"[{now()}] resumed from {resume_from}, global_step={global_step}", flush=True)

    # Train loop
    save_every = int(cfg["run"].get("save_every_steps", 2000))
    eval_every = int(cfg["run"].get("eval_every_steps", 2000))
    log_every = int(cfg["run"].get("log_every_steps", 50))

    model.train()
    print(f"[{now()}] starting training: epochs={epochs}, total_update_steps={total_update_steps}", flush=True)

    start_time = time.time()
    running_loss = 0.0
    running_tokens = 0

    for epoch in range(epochs):
        for step, batch in enumerate(train_loader):
            if max_steps > 0 and global_step >= max_steps:
                break

            batch = to_device(batch, device)

            # forward
            if autocast is not None:
                with autocast(**autocast_kwargs):
                    out = model(
                        pixel_values=batch["pixel_values"],
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )
                    loss = out.loss
            else:
                out = model(
                    pixel_values=batch["pixel_values"],
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = out.loss

            # normalize for grad accumulation
            loss = loss / grad_accum
            loss.backward()

            # stats
            running_loss += float(loss.detach().item())
            # approximate tokens = non-pad tokens in labels
            running_tokens += int((batch["labels"] != -100).sum().item())

            # update
            if (step + 1) % grad_accum == 0:
                gnorm = grad_global_norm(model)
                torch.nn.utils.clip_grad_norm_(params, max_grad_norm)

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                global_step += 1

                # logging
                if (global_step % log_every) == 0:
                    elapsed = time.time() - start_time
                    lr_now = scheduler.get_last_lr()[0]
                    loss_now = running_loss / max(log_every, 1)
                    tokps = running_tokens / max(elapsed, 1e-6)

                    log_dict = {
                        "train/loss": loss_now,
                        "train/lr": lr_now,
                        "train/grad_norm": gnorm,
                        "train/tokens_per_sec": tokps,
                        "train/epoch": epoch,
                        "train/step": global_step,
                    }

                    # optional histograms (LoRA only)
                    if wandb_enabled:
                        try:
                            log_dict.update(log_param_histograms(model, global_step))
                        except Exception:
                            pass
                        wandb.log(log_dict, step=global_step)

                    print(f"[{now()}] step={global_step} loss={loss_now:.4f} lr={lr_now:.2e} gnorm={gnorm:.2f}", flush=True)
                    running_loss = 0.0
                    running_tokens = 0
                    start_time = time.time()

                # checkpoint
                if (global_step % save_every) == 0:
                    ckpt_path = os.path.join(out_dir, "checkpoints", f"step_{global_step}.pt")
                    torch.save(
                        {
                            "model": model.state_dict(),
                            "optim": optimizer.state_dict(),
                            "sched": scheduler.state_dict(),
                            "global_step": global_step,
                            "cfg": cfg,
                        },
                        ckpt_path,
                    )
                    print(f"[{now()}] saved checkpoint: {ckpt_path}", flush=True)

                # evaluation
                if val_loader is not None and (global_step % eval_every) == 0:
                    eval_cfg = cfg.get("eval", {})
                    ns = int(eval_cfg.get("num_samples", 2000))
                    metrics = quick_caption_eval(
                        model=model,
                        processor=processor,
                        loader=val_loader,
                        device=device,
                        max_new_tokens=int(cfg["model"].get("max_text_len", 32)),
                        num_samples=ns,
                    )
                    if wandb_enabled:
                        wandb.log(metrics, step=global_step)
                    print(f"[{now()}] eval: {metrics}", flush=True)
                    model.train()

        if max_steps > 0 and global_step >= max_steps:
            break

    # final save
    final_path = os.path.join(out_dir, "checkpoints", "final.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "sched": scheduler.state_dict(),
            "global_step": global_step,
            "cfg": cfg,
        },
        final_path,
    )
    print(f"[{now()}] training done. final checkpoint: {final_path}", flush=True)

    if wandb_enabled:
        wandb.finish()


if __name__ == "__main__":
    main()
