#!/usr/bin/env python3
import os
import json
import time
import argparse
import random
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import wandb
from pycocotools.coco import COCO

from lavis.models import load_model_and_preprocess
from torchmetrics.classification import MultilabelAveragePrecision


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def is_main_process() -> bool:
    # We are not doing DDP here; kept for future extension
    return True


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# -----------------------------
# COCO Dataset (multi-label)
# -----------------------------
class COCOMultiLabel(Dataset):
    def __init__(self, img_root: str, ann_path: str, vis_processor):
        self.coco = COCO(ann_path)
        self.img_root = img_root
        self.vis_processor = vis_processor

        self.img_ids = sorted(self.coco.getImgIds())
        self.cat_ids = sorted(self.coco.getCatIds())
        self.catid_to_idx = {cid: i for i, cid in enumerate(self.cat_ids)}

        cats = self.coco.loadCats(self.cat_ids)
        # COCO class names in official order of cat_ids
        self.class_names = [c["name"] for c in cats]

        # cache labels
        self.labels = {}
        for img_id in self.img_ids:
            ann_ids = self.coco.getAnnIds(imgIds=[img_id])
            anns = self.coco.loadAnns(ann_ids)
            y = np.zeros(len(self.cat_ids), dtype=np.float32)
            for a in anns:
                y[self.catid_to_idx[a["category_id"]]] = 1.0
            self.labels[img_id] = y

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        path = os.path.join(self.img_root, img_info["file_name"])
        image = Image.open(path).convert("RGB")
        image = self.vis_processor(image)
        label = self.labels[img_id]
        return image, label, img_id


# -----------------------------
# VLM -> pooled feature -> head
# -----------------------------
class VLClassifier(nn.Module):
    """
    Head-only finetuning by default.
    Uses LAVIS load_model_and_preprocess for BLIP-2 + BLIP.
    """
    def __init__(self, model_name: str, model_type: str, num_classes: int, device: str):
        super().__init__()
        self.device = device
        self.num_classes = num_classes

        self.model, self.vis_processors, self.txt_processors = load_model_and_preprocess(
            name=model_name, model_type=model_type, is_eval=False, device=device)

        # Freeze BLIP-2 backbone (head-only fine-tuning)
        for p in self.model.parameters():
            p.requires_grad = False

        # ---- VISUAL FEATURE DIMENSION ----
        vision_dim = self.model.visual_encoder.num_features

        # ---- CLASSIFICATION HEAD ----
        self.classifier = nn.Linear(vision_dim, num_classes).to(device)



    def _build_head_if_needed(self, feat_dim: int):
        if self.head is None:
            self.head = nn.Linear(feat_dim, self.num_classes).to(self.device)

    @torch.no_grad()
    def extract_pooled_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B,3,H,W] float32 from DataLoader returns pooled visual features [B,D] in float32"""
        # Ensure images are on the same device
        images = images.to(self.device, non_blocking=True)

        # Match dtype of vision encoder params (usually fp16 on GPU)
        ve_dtype = next(self.model.visual_encoder.parameters()).dtype
        images = images.to(dtype=ve_dtype)

        # Run vision encoder under autocast as extra safety
        with torch.no_grad():
            with torch.cuda.amp.autocast(enabled=images.is_cuda):
                vision_embeds = self.model.visual_encoder(images)  # [B, N, D] or similar
    
                # pool tokens (mean pooling)
                pooled = vision_embeds.mean(dim=1)
    
        return pooled.float()  # head expects fp32

        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.extract_pooled_image_features(images)  # fp32
        logits = self.classifier(feats)                    # fp32
        return logits


# -----------------------------
# Zero-shot multi-label scoring
# -----------------------------
@torch.no_grad()
def zeroshot_eval(
    model_wrapper: VLClassifier,
    class_names: List[str],
    prompt_template: str,
    loader: DataLoader,
    threshold: float,
    device: str
) -> Dict[str, float]:
    """
    Zero-shot multi-label using image-text similarity:
    - Build prompts: template.format(class)
    - Encode text
    - Encode image
    - Similarity -> sigmoid -> threshold
    NOTE: Text encoding API differs by model; we implement a robust fallback:
          use model_wrapper.model to compute text features if available.
    """
    model_wrapper.eval()

    # Build text prompts
    prompts = [prompt_template.format(c) for c in class_names]

    # Text features: try model.encode_text or extract_features(mode="text")
    # We’ll compute one time.
    text_feats = None

    # (A) try: model_wrapper.model.extract_features(mode="text")
    try:
        txt_inputs = [model_wrapper.txt_processors["eval"](p) for p in prompts]
        # Many LAVIS models accept {"text_input": [...]} for text mode
        out_t = model_wrapper.model.extract_features({"text_input": txt_inputs}, mode="text")
        for k in ["text_embeds", "text_embed", "embeds"]:
            if hasattr(out_t, k) and getattr(out_t, k) is not None:
                text_feats = getattr(out_t, k)
                break
        if text_feats is None and isinstance(out_t, dict):
            for k in ["text_embeds", "text_embed", "embeds"]:
                if k in out_t and out_t[k] is not None:
                    text_feats = out_t[k]
                    break
        if text_feats is not None:
            if text_feats.dim() == 3:
                text_feats = text_feats.mean(dim=1)
            text_feats = F.normalize(text_feats, dim=-1)
    except Exception:
        text_feats = None

    if text_feats is None:
        raise RuntimeError(
            "Zero-shot text feature extraction failed for this LAVIS model. "
            "Run fine-tuning mode or adapt zeroshot_eval() to your exact model APIs."
        )

    num_classes = len(class_names)
    ap = MultilabelAveragePrecision(num_labels=num_classes, average="macro").to(device)

    all_loss = []
    for images, labels, _ in loader:
        images = images.to(device)
        labels = torch.tensor(labels, device=device)

        img_feats = model_wrapper.extract_pooled_image_features(images)
        img_feats = F.normalize(img_feats, dim=-1)

        # similarity [B, C]
        sim = img_feats @ text_feats.T
        probs = torch.sigmoid(sim)
        pred = (probs >= threshold).int()

        # pseudo-loss just to log something (not used)
        loss = F.binary_cross_entropy(probs, labels)
        all_loss.append(loss.item())

        ap.update(probs, labels.int())

    return {
        "val/zeroshot_loss": float(np.mean(all_loss)) if all_loss else 0.0,
        "val/zeroshot_mAP": float(ap.compute().item())
    }


# -----------------------------
# Train / Eval
# -----------------------------
@dataclass
class TrainConfig:
    coco_root: str
    ann_train: str
    ann_val: str
    model_name: str
    model_type: str
    epochs: int
    batch_size: int
    grad_accum: int
    lr: float
    weight_decay: float
    optimizer: str
    num_workers: int
    threshold: float
    prompt_template: str
    amp: bool
    seed: int
    run_zeroshot: bool
    finetune_mode: str
    output_dir: str


def build_optimizer(name: str, params, lr: float, weight_decay: float):
    name = name.lower()
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)
    raise ValueError(f"Unknown optimizer: {name}")


@torch.no_grad()
def evaluate(model: VLClassifier, loader: DataLoader, device: str, num_classes: int) -> Tuple[float, float]:
    model.eval()
    ap = MultilabelAveragePrecision(num_labels=num_classes, average="macro").to(device)
    losses = []

    for images, labels, _ in loader:
        images = images.to(device)
        labels = torch.tensor(labels, device=device)
        logits = model(images)
        loss = F.binary_cross_entropy_with_logits(logits, labels)
        losses.append(loss.item())
        ap.update(torch.sigmoid(logits), labels.int())

    return float(np.mean(losses) if losses else 0.0), float(ap.compute().item())


def train(cfg: TrainConfig):
    device = get_device()
    set_seed(cfg.seed)

    os.makedirs(cfg.output_dir, exist_ok=True)

    # W&B init
    wandb.init(
        project="coco-blip2-classification",
        config=cfg.__dict__,
        settings=wandb.Settings(start_method="thread")
    )
    wb = wandb.config

    # Load model + preprocess
    model = VLClassifier(
        model_name=cfg.model_name,
        model_type=cfg.model_type,
        num_classes=80,
        device=device
    )

    # Build datasets
    # Resolve COCO annotation paths (override defaults)
    if cfg.ann_train is None:
        cfg.ann_train = os.path.join(
        cfg.coco_root,
        "annotations",
        "instances_train2017.json",
    )

    if cfg.ann_val is None:
        cfg.ann_val = os.path.join(
        cfg.coco_root,
        "annotations",
        "instances_val2017.json",
    )

    train_ds = COCOMultiLabel(
    img_root=os.path.join(cfg.coco_root, "images", "train2017"),
    ann_path=cfg.ann_train,
    vis_processor=model.vis_processors["train"],
    )

    val_ds = COCOMultiLabel(
    img_root=os.path.join(cfg.coco_root, "images", "val2017"),
    ann_path=cfg.ann_val,
    vis_processor=model.vis_processors["eval"],
    )

    train_loader = DataLoader(
    train_ds,
    batch_size=cfg.batch_size,
    shuffle=True,
    num_workers=cfg.num_workers,
    pin_memory=True,
    drop_last=True,
    )

    val_loader = DataLoader(
    val_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True,
    )

    # Optional zero-shot eval before training
    if cfg.run_zeroshot:
        z = zeroshot_eval(
            model_wrapper=model,
            class_names=train_ds.class_names,
            prompt_template=cfg.prompt_template,
            loader=val_loader,
            threshold=cfg.threshold,
            device=device
        )
        wandb.log(z)

    # Build head now so optimizer sees its params
    # (do one tiny forward)
    val_loader = DataLoader(
    val_ds,
    batch_size=cfg.batch_size,
    shuffle=False,
    num_workers=cfg.num_workers,
    pin_memory=True
    )

    with torch.no_grad():
        x0, _, _ = next(iter(val_loader))
        x0 = x0.to(device)
        _ = model(x0)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = build_optimizer(cfg.optimizer, params, cfg.lr, cfg.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    best_map = -1.0
    best_path = os.path.join(cfg.output_dir, "best_head.pt")

    global_step = 0
    for epoch in range(int(cfg.epochs)):
        model.train()
        running = 0.0
        opt.zero_grad(set_to_none=True)

        for it, (images, labels, _) in enumerate(train_loader):
            images = images.to(device)
            labels = torch.tensor(labels, device=device)

            with torch.cuda.amp.autocast(enabled=cfg.amp):
                logits = model(images)
                loss = F.binary_cross_entropy_with_logits(logits, labels)
                loss = loss / max(1, cfg.grad_accum)

            scaler.scale(loss).backward()

            if (it + 1) % cfg.grad_accum == 0:
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

            running += loss.item() * max(1, cfg.grad_accum)
            global_step += 1

            if global_step % 50 == 0:
                wandb.log({"train/loss_step": running / (it + 1), "step": global_step})

        train_loss = running / max(1, len(train_loader))
        val_loss, val_map = evaluate(model, val_loader, device=device, num_classes=80)

        wandb.log({
            "epoch": epoch,
            "train/loss": train_loss,
            "val/loss": val_loss,
            "val/mAP": val_map
        })

        if val_map > best_map:
            best_map = val_map
            ckpt = {
                "head_state_dict": model.head.state_dict(),
                "model_name": cfg.model_name,
                "model_type": cfg.model_type,
                "config": cfg.__dict__,
                "best_val_mAP": best_map,
            }
            torch.save(ckpt, best_path)
            wandb.save(best_path)

    # Save artifact
    if is_main_process():
        artifact = wandb.Artifact(
            name=f"{cfg.model_name}-{cfg.model_type}-coco-head",
            type="model"
        )
        artifact.add_file(best_path)
        wandb.log_artifact(artifact)

    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument( "--coco_root",type=str,required=True, help="Path to COCO root containing images/ and annotations/")

    p.add_argument("--ann_train",type=str,default=None,help="Path to instances_train2017.json (optional, derived from coco_root if not set)")

    p.add_argument("--ann_val",type=str,default=None,help="Path to instances_val2017.json (optional, derived from coco_root if not set)")


    p.add_argument("--model_name", type=str, required=True, help="blip2 or blip")
    p.add_argument("--model_type", type=str, required=True, help="e.g., pretrain_vitg / pretrain_vitl / large")

    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--optimizer", type=str, default="adamw")
    p.add_argument("--num_workers", type=int, default=4)

    p.add_argument("--threshold", type=float, default=0.4)
    p.add_argument("--prompt_template", type=str, default="a photo of a {}")
    p.add_argument("--amp", action="store_true")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--run_zeroshot", action="store_true")
    p.add_argument("--finetune_mode", type=str, default="head_only")

    p.add_argument("--output_dir", type=str, default=os.path.join("checkpoints"))

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    cfg = TrainConfig(
        coco_root=args.coco_root,        ann_train=args.ann_train,
        ann_val=args.ann_val,
        model_name=args.model_name,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        num_workers=args.num_workers,
        threshold=args.threshold,
        prompt_template=args.prompt_template,
        amp=bool(args.amp),
        seed=args.seed,
        run_zeroshot=bool(args.run_zeroshot),
        finetune_mode=args.finetune_mode,
        output_dir=args.output_dir,
    )
    train(cfg)
