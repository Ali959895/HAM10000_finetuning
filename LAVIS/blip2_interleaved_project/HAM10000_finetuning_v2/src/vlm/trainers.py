# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from tqdm import tqdm


# ------------------------
# Utilities
# ------------------------
def get_trainable_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    """Return only trainable parameters (dramatically smaller checkpoints)."""
    trainable = {}
    for name, p in model.named_parameters():
        if p.requires_grad:
            trainable[name] = p.detach().cpu()
    return trainable


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class FocalLoss(nn.Module):
    """Multiclass focal loss for logits."""
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None, label_smoothing: float = 0.0):
        super().__init__()
        self.gamma = float(gamma)
        self.register_buffer("weight", weight if weight is not None else None)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, target, weight=self.weight, reduction="none", label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()


def build_loss(cfg: dict, num_classes: int, device: torch.device) -> nn.Module:
    """
    cfg example:
      train:
        loss:
          name: ce          # ce | focal
          label_smoothing: 0.0
          focal_gamma: 2.0
        class_weights: balanced   # null | balanced | [..C..]
    """
    train_cfg = cfg.get("train", {})
    loss_cfg = train_cfg.get("loss", {}) or {}
    name = (loss_cfg.get("name", "ce") or "ce").lower()

    cw = train_cfg.get("class_weights", None)
    weight = None
    if isinstance(cw, (list, tuple)) and len(cw) == num_classes:
        weight = torch.tensor(cw, dtype=torch.float32, device=device)

    label_smoothing = float(loss_cfg.get("label_smoothing", 0.0) or 0.0)

    if name in ("ce", "cross_entropy", "cross-entropy"):
        return nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
    if name in ("focal", "focal_loss"):
        gamma = float(loss_cfg.get("focal_gamma", 2.0) or 2.0)
        return FocalLoss(gamma=gamma, weight=weight, label_smoothing=label_smoothing)

    raise ValueError(f"Unknown loss.name: {name}")


# ------------------------
# Evaluation
# ------------------------
@torch.no_grad()
def evaluate_multiclass(
    model: nn.Module,
    loader,
    device,
    amp: bool = False,
    criterion: Optional[nn.Module] = None,
) -> Dict[str, float]:
    model.eval()

    all_probs = []
    all_pred = []
    all_true = []
    losses = []

    for batch in tqdm(loader, desc="eval", leave=False):
        x = batch["image"].to(device, non_blocking=True)
        y = batch["label"].to(device, non_blocking=True)

        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            if criterion is not None:
                losses.append(float(criterion(logits, y).detach().cpu()))

        probs = torch.softmax(logits.float(), dim=-1)
        pred = probs.argmax(dim=-1)

        all_probs.append(_to_numpy(probs))
        all_pred.append(_to_numpy(pred))
        all_true.append(_to_numpy(y))

    probs = np.concatenate(all_probs, axis=0)
    pred = np.concatenate(all_pred, axis=0)
    true = np.concatenate(all_true, axis=0)

    metrics = {
        "acc": float(accuracy_score(true, pred)),
        "bacc": float(balanced_accuracy_score(true, pred)),
        "f1_macro": float(f1_score(true, pred, average="macro")),
        "f1_weighted": float(f1_score(true, pred, average="weighted")),
        "log_loss": float(log_loss(true, probs, labels=list(range(probs.shape[1])))),
    }

    # AUROC OVR macro (guard for small class counts)
    try:
        metrics["auroc_ovr_macro"] = float(roc_auc_score(true, probs, multi_class="ovr", average="macro"))
    except Exception:
        metrics["auroc_ovr_macro"] = float("nan")

    if losses:
        metrics["loss"] = float(np.mean(losses))

    return metrics


# ------------------------
# Training
# ------------------------
def train_multiclass(model, train_loader, val_loader, cfg, run_dir, wandb=None):
    device = torch.device(cfg.get("device", "cuda"))
    model.to(device)

    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 10))
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    amp = bool(train_cfg.get("amp", True))
    log_every = int(train_cfg.get("log_every", 50))
    grad_clip = float(train_cfg.get("grad_clip", 0.0) or 0.0)

    # selection for best checkpoint
    select_metric = str(train_cfg.get("select_metric", "f1_macro"))
    select_mode = str(train_cfg.get("select_mode", "max")).lower()  # max|min

    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")

    # Optimizer over trainable parameters only
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)

    scaler = torch.cuda.amp.GradScaler(enabled=amp)

    # loss
    num_classes = int(cfg.get("num_classes", train_cfg.get("num_classes", 7)))
    criterion = build_loss(cfg, num_classes=num_classes, device=device)

    best_score = None
    best_path = os.path.join(run_dir, "best_trainable.pt")
    last_path = os.path.join(run_dir, "last_trainable.pt")

    global_step = 0

    for epoch in range(epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"train {epoch}", leave=False)
        running = 0.0
        n_seen = 0

        for batch in pbar:
            global_step += 1
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
                loss = criterion(logits, y)

            scaler.scale(loss).backward()
            if grad_clip > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(params, grad_clip)

            scaler.step(opt)
            scaler.update()

            bs = int(x.size(0))
            running += float(loss.detach().cpu()) * bs
            n_seen += bs

            if log_every > 0 and (global_step % log_every == 0):
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
                if wandb:
                    wandb.log({"train/loss_step": float(loss.item())}, step=global_step)

        train_loss = running / max(1, n_seen)

        # validation
        val_metrics = evaluate_multiclass(model, val_loader, device=device, amp=amp, criterion=criterion)
        val_loss = float(val_metrics.get("loss", float("nan")))

        # pick best
        score = float(val_metrics.get(select_metric, float("nan")))
        if best_score is None:
            is_best = True
        else:
            is_best = (score > best_score) if select_mode == "max" else (score < best_score)

        if is_best:
            best_score = score
            torch.save(
                {"trainable": get_trainable_state_dict(model), "cfg": cfg, "epoch": epoch, "step": global_step},
                best_path,
            )

        # always save last (trainable only)
        torch.save(
            {"trainable": get_trainable_state_dict(model), "cfg": cfg, "epoch": epoch, "step": global_step},
            last_path,
        )

        row = {
            "split": "val",
            "epoch": epoch,
            "step": global_step,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "best_metric": select_metric,
            "best_score": float(best_score) if best_score is not None else None,
            "saved_best": bool(is_best),
            **{k: float(v) for k, v in val_metrics.items()},
        }

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        # Console summary
        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} {select_metric}={score:.4f} "
            f"{'(BEST)' if is_best else ''}"
        )

        if wandb:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/loss_epoch": float(train_loss),
                    "val/loss": float(val_loss),
                    **{f"val/{k}": float(v) for k, v in val_metrics.items()},
                },
                step=global_step,
            )

    return {"best_path": best_path, "best_score": best_score}
