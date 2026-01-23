# -*- coding: utf-8 -*-
"""
HAM10000 finetuning runner (multiclass).

Modes:
  - train_multiclass: train + validate each epoch, save best & last (trainable-only checkpoints)
  - eval_multiclass : evaluate a checkpoint on val or test split
  - zeroshot_multiclass: (optional) placeholder (use CLIP baseline if you want)

This script is intentionally defensive:
- Handles broken/partial configs
- Avoids huge checkpoints by saving trainable parameters only
- Writes outputs to /scratch by default (Compute Canada quota-safe)

Expected CSV format (train/val/test):
  - one row per sample
  - image column: one of ["image", "image_path", "path", "file", "img", "image_id"]
  - label column: one of ["label", "dx", "target", "y"]

If image_id is provided, it will be resolved as:
  img_root / image_id.(jpg|png|jpeg) if the extension is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import time
import copy
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from vlm.models.blip2_classifier import Blip2Classifier
from vlm.models.clip_classifier import ClipClassifier
from vlm.trainers import evaluate_multiclass, train_multiclass


# ------------------------
# Config utilities
# ------------------------
def load_yaml(path: str) -> Dict[str, Any]:
    import yaml
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}
    if not isinstance(cfg, dict):
        return {}
    return cfg


def safe_write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        with open(path, "w", encoding="utf-8") as f:
            f.write(text)
    except OSError as e:
        # Common on CC if HOME quota exceeded
        print(f"[WARN] Could not write {path}: {e}")


def resolve_output_root(cfg: Dict[str, Any]) -> str:
    out = cfg.get("output_root") or cfg.get("run", {}).get("output_root")
    if out:
        return os.path.expandvars(out)
    user = os.environ.get("USER", "user")
    return f"/scratch/{user}/ham_runs"


# ------------------------
# Data
# ------------------------
def _pick_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _resolve_image_path(img_root: str, v: str) -> str:
    v = str(v)
    if os.path.isabs(v) and os.path.exists(v):
        return v
    p = os.path.join(img_root, v)
    if os.path.exists(p):
        return p
    # if extension missing, try common ones
    base, ext = os.path.splitext(p)
    if ext == "":
        for e in (".jpg", ".jpeg", ".png"):
            if os.path.exists(base + e):
                return base + e
    return p  # best effort


class CsvImageDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        img_root: str,
        classes: List[str],
        transform=None,
        image_col: Optional[str] = None,
        label_col: Optional[str] = None,
        label_map: Optional[Dict[str, str]] = None,
    ):
        self.csv_path = csv_path
        self.img_root = img_root
        self.transform = transform
        df = pd.read_csv(csv_path)

        self.image_col = image_col or _pick_col(df, ["image", "image_path", "path", "file", "img", "image_id"])
        self.label_col = label_col or _pick_col(df, ["label", "dx", "target", "y"])
        if self.image_col is None or self.label_col is None:
            raise ValueError(
                f"CSV {csv_path} must contain image+label columns. "
                f"Found columns: {list(df.columns)}"
            )

        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        self.items: List[Tuple[str, int]] = []

        for _, row in df.iterrows():
            img = _resolve_image_path(img_root, row[self.image_col])
            lab = row[self.label_col]
            lab = str(lab)
            if label_map is not None:
                lab = str(label_map.get(lab, lab))
            if lab not in self.class_to_idx:
                # allow unseen labels if user passes explicit classes
                continue
            self.items.append((img, self.class_to_idx[lab]))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        path, y = self.items[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"image": img, "label": torch.tensor(y, dtype=torch.long), "path": path}


def build_transforms(cfg: Dict[str, Any], train: bool) -> transforms.Compose:
    data_cfg = cfg.get("data", {}) or {}
    aug = data_cfg.get("augment", {}) or {}

    image_size = int(data_cfg.get("image_size", 224))
    mean = aug.get("mean", [0.485, 0.456, 0.406])
    std = aug.get("std", [0.229, 0.224, 0.225])

    if not train:
        return transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.15)),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

    # "standard" augmentation set for medical skin-lesion classification
    # (kept conservative; you can increase strength in config)
    rr_scale = aug.get("random_resized_crop_scale", [0.8, 1.0])
    color_jitter = aug.get("color_jitter", [0.2, 0.2, 0.2, 0.05])
    rot = float(aug.get("random_rotation", 20))
    hflip_p = float(aug.get("hflip_p", 0.5))
    vflip_p = float(aug.get("vflip_p", 0.5))
    re_p = float(aug.get("random_erasing_p", 0.1))

    tfs = [
        transforms.RandomResizedCrop(image_size, scale=tuple(rr_scale)),
        transforms.RandomHorizontalFlip(p=hflip_p),
        transforms.RandomVerticalFlip(p=vflip_p),
        transforms.RandomRotation(rot),
        transforms.ColorJitter(*color_jitter),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
    if re_p > 0:
        tfs.append(transforms.RandomErasing(p=re_p, scale=(0.02, 0.1), ratio=(0.3, 3.3), value="random"))
    return transforms.Compose(tfs)


def compute_balanced_class_weights(train_ds: CsvImageDataset, num_classes: int) -> List[float]:
    counts = np.zeros(num_classes, dtype=np.int64)
    for _, y in train_ds.items:
        counts[y] += 1
    counts = np.maximum(counts, 1)
    inv = 1.0 / counts.astype(np.float64)
    w = inv / inv.mean()
    return w.astype(np.float32).tolist()


def build_ham_dataloaders(cfg: Dict[str, Any]):
    data_cfg = cfg.get("data", {}) or {}
    train_csv = os.path.expandvars(data_cfg["train_csv"])
    val_csv = os.path.expandvars(data_cfg["val_csv"])
    test_csv = os.path.expandvars(data_cfg.get("test_csv", "")) if data_cfg.get("test_csv") else None
    img_root = os.path.expandvars(data_cfg["img_root"])

    task_cfg = cfg.get("task", {}) or {}
    label_map = task_cfg.get("label_map", None)

    # classes inferred from train CSV unless explicitly provided
    classes = data_cfg.get("classes", None)
    if not classes:
        df = pd.read_csv(train_csv)
        label_col = _pick_col(df, ["label", "dx", "target", "y"])
        task_cfg = cfg.get("task", {}) or {}
        label_map = task_cfg.get("label_map", None)
        if isinstance(label_map, dict) and label_map:
            df[label_col] = df[label_col].astype(str).map(lambda v: str(label_map.get(str(v), str(v))))
        if label_col is None:
            raise ValueError(f"Cannot infer classes; label column not found in {train_csv}")
        classes = sorted(df[label_col].astype(str).unique().tolist())

    train_tf = build_transforms(cfg, train=True)
    eval_tf = build_transforms(cfg, train=False)

    train_ds = CsvImageDataset(train_csv, img_root, classes, transform=train_tf, label_map=label_map)
    val_ds = CsvImageDataset(val_csv, img_root, classes, transform=eval_tf, label_map=label_map)
    test_ds = CsvImageDataset(test_csv, img_root, classes, transform=eval_tf, label_map=label_map) if test_csv else None

    bs = int(data_cfg.get("batch_size", 16))
    num_workers = int(data_cfg.get("num_workers", 1))
    pin_memory = bool(data_cfg.get("pin_memory", True))

    train_dl = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=num_workers, pin_memory=pin_memory)
    val_dl = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin_memory)
    test_dl = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=num_workers, pin_memory=pin_memory) if test_ds else None

    return train_dl, val_dl, test_dl, classes, train_ds


# ------------------------
# Model
# ------------------------
def build_model(cfg: Dict[str, Any], num_classes: int) -> torch.nn.Module:
    model_cfg = cfg.get("model", {}) or {}

    arch = str(model_cfg.get("arch", "blip2_opt")).lower()
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    pooling = str(model_cfg.get("pooling", "mean"))
    head_hidden = int(model_cfg.get("head_hidden", 0) or 0)
    activation = str(model_cfg.get("activation", "gelu"))
    dropout = float(model_cfg.get("dropout", 0.0) or 0.0)

    if arch in ("clip",):
        clip_cfg = model_cfg.get("clip", {}) or {}
        return ClipClassifier(
            num_classes=num_classes,
            device=device,
            model_name=str(clip_cfg.get("model_name", "ViT-L-14")),
            pretrained=str(clip_cfg.get("pretrained", "openai")),
            pooling=pooling,
            head_hidden=head_hidden,
            activation=activation,
            dropout=dropout,
        )

    # default: use LAVIS backbone via Blip2Classifier wrapper
    lavis_name = str(model_cfg.get("lavis_name", "blip2_opt"))
    model_type = str(model_cfg.get("model_type", "pretrain_opt2.7b"))

    return Blip2Classifier(
        num_classes=num_classes,
        lavis_name=lavis_name,
        model_type=model_type,
        model_cfg=model_cfg,
        device=device,
        train_qformer=bool(model_cfg.get("train_qformer", False)),
        train_vision=bool(model_cfg.get("train_vision", False)),
        unfreeze_vision_last_n=int(model_cfg.get("unfreeze_vision_last_n", 0) or 0),
        pooling=pooling,
        head_hidden=head_hidden,
        activation=activation,
        dropout=dropout,
    )


def load_trainable_checkpoint(model: torch.nn.Module, ckpt_path: str) -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("trainable", ckpt.get("model", ckpt))
    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[INFO] Loaded trainable ckpt. missing={len(missing)} unexpected={len(unexpected)}")
    return ckpt


# ------------------------
# Main
# ------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-c", "--config", required=True)
    ap.add_argument("--mode", required=True, choices=["train_multiclass", "eval_multiclass", "benchmark_multiclass", "zeroshot_multiclass"])
    args = ap.parse_args()

    cfg = load_yaml(args.config)

    # resolve output root/run dir
    out_root = resolve_output_root(cfg)
    run_name = (cfg.get("run", {}) or {}).get("name", "ham10000")
    run_id = time.strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(out_root, f"{run_name}_{run_id}")
    cfg["output_root"] = out_root
    cfg["run_dir"] = run_dir

    os.makedirs(run_dir, exist_ok=True)
    import yaml
    safe_write_text(os.path.join(run_dir, "config_resolved.yaml"), yaml.safe_dump(cfg, sort_keys=False))

    # data
    train_dl, val_dl, test_dl, classes, train_ds = build_ham_dataloaders(cfg)
    num_classes = len(classes)
    cfg["num_classes"] = int(num_classes)
    cfg["num_classes"] = num_classes

    # class weights (optional)
    train_cfg = cfg.get("train", {}) or {}
    cw = train_cfg.get("class_weights", None)
    if isinstance(cw, str) and cw.lower() == "balanced":
        weights = compute_balanced_class_weights(train_ds, num_classes)
        cfg.setdefault("train", {})["class_weights"] = weights
        safe_write_text(os.path.join(run_dir, "class_weights.json"), json.dumps(weights, indent=2))
        print("[INFO] Using balanced class weights.")

    # save class list
    safe_write_text(os.path.join(run_dir, "classes.json"), json.dumps(classes, indent=2))

    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    if args.mode == "train_multiclass":
        model = build_model(cfg, num_classes=num_classes)
        train_multiclass(model, train_dl, val_dl, cfg, run_dir, wandb=None)
        return

    if args.mode == "benchmark_multiclass":
        bench_cfg = cfg.get("benchmark", {}) or {}
        models = bench_cfg.get("models", []) or []
        if not models:
            raise KeyError("cfg.benchmark.models is empty; nothing to benchmark")

        bench_dir = os.path.join(run_dir, "benchmark_" + time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(bench_dir, exist_ok=True)

        results = []

        for i, mcfg in enumerate(models):
            mname = str(mcfg.get("name", f"model_{i}"))
            print(f"\n[INFO] ===== Benchmark {i+1}/{len(models)}: {mname} =====")

            sub_cfg = copy.deepcopy(cfg)
            sub_cfg.setdefault("model", {})
            # override model keys from entry
            for k, v in (mcfg or {}).items():
                if k == "name":
                    continue
                sub_cfg["model"][k] = v

            sub_run_dir = os.path.join(bench_dir, re.sub(r"[^A-Za-z0-9_.-]+", "_", mname))
            os.makedirs(sub_run_dir, exist_ok=True)

            import yaml
            safe_write_text(os.path.join(sub_run_dir, "config_resolved.yaml"), yaml.safe_dump(sub_cfg, sort_keys=False))

            model = build_model(sub_cfg, num_classes=num_classes)
            train_multiclass(model, train_dl, val_dl, sub_cfg, sub_run_dir, wandb=None)

            # evaluate best on val (and test if available)
            best_ckpt = os.path.join(sub_run_dir, "best_trainable.pt")
            model = build_model(sub_cfg, num_classes=num_classes)
            load_trainable_checkpoint(model, best_ckpt)
            model = model.to(torch.device(device))

            val_metrics = evaluate_multiclass(model, val_dl, device=torch.device(device), amp=bool((sub_cfg.get("train", {}) or {}).get("amp", True)))
            test_metrics = None
            if test_dl is not None:
                test_metrics = evaluate_multiclass(model, test_dl, device=torch.device(device), amp=bool((sub_cfg.get("train", {}) or {}).get("amp", True)))

            row = {"name": mname}
            for k, v in val_metrics.items():
                row[f"val_{k}"] = v
            if isinstance(test_metrics, dict):
                for k, v in test_metrics.items():
                    row[f"test_{k}"] = v

            results.append(row)
            safe_write_text(os.path.join(sub_run_dir, "best_val_metrics.json"), json.dumps(val_metrics, indent=2))
            if isinstance(test_metrics, dict):
                safe_write_text(os.path.join(sub_run_dir, "best_test_metrics.json"), json.dumps(test_metrics, indent=2))

        # save benchmark summary
        try:
            import pandas as pd
            df = pd.DataFrame(results)
            df.to_csv(os.path.join(bench_dir, "benchmark_summary.csv"), index=False)
            safe_write_text(os.path.join(bench_dir, "benchmark_summary.json"), json.dumps(results, indent=2))
            print("\n[INFO] Benchmark summary saved to:", bench_dir)
            print(df)
        except Exception as e:
            safe_write_text(os.path.join(bench_dir, "benchmark_summary.json"), json.dumps(results, indent=2))
            print(f"[WARN] Could not write CSV summary: {e}")
        return

    if args.mode == "eval_multiclass":
        model = build_model(cfg, num_classes=num_classes)
        eval_cfg = cfg.get("eval", {}) or {}
        ckpt_path = eval_cfg.get("checkpoint", None)
        if not ckpt_path:
            raise KeyError("cfg.eval.checkpoint is required for eval_multiclass")
        ckpt_path = os.path.expandvars(ckpt_path)
        load_trainable_checkpoint(model, ckpt_path)
        model = model.to(torch.device(device))
        model.eval()
        split = str(eval_cfg.get("split", "val")).lower()
        loader = val_dl if split == "val" else (test_dl if test_dl is not None else val_dl)
        metrics = evaluate_multiclass(model, loader, device=torch.device(device), amp=bool(train_cfg.get("amp", True)))
        safe_write_text(os.path.join(run_dir, f"eval_{split}.json"), json.dumps(metrics, indent=2))
        print("[EVAL]", split, metrics)
        return

    # zeroshot placeholder
    raise NotImplementedError(
        "zeroshot_multiclass is not wired in this clean runner. "
        "If you want, I can add an OpenCLIP baseline here."
    )


if __name__ == "__main__":
    main()
