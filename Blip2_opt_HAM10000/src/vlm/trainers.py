# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import numpy as np
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, log_loss, confusion_matrix
)
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, log_loss, confusion_matrix
)

# Optional text metrics (safe: code will skip if not installed / no texts)
try:
    import evaluate
except Exception:
    evaluate = None

try:
    from bert_score import score as bertscore
except Exception:
    bertscore = None


def safe_write_json(path: str, obj) -> None:
    """Best-effort JSON writer."""
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2)
    except OSError as e:
        print(f"[WARN] Could not write {path}: {e}")

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
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

# Optional text metrics
try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _HAS_NLTK = True
except Exception:
    _HAS_NLTK = False

def _rouge_l_f1(pred: str, ref: str) -> float:
    """
    Simple ROUGE-L F1 based on LCS.
    No external deps.
    """
    pred_tokens = pred.split()
    ref_tokens = ref.split()
    if len(pred_tokens) == 0 or len(ref_tokens) == 0:
        return 0.0

    # LCS DP
    m, n = len(ref_tokens), len(pred_tokens)
    dp = [[0]*(n+1) for _ in range(m+1)]
    for i in range(m):
        for j in range(n):
            if ref_tokens[i] == pred_tokens[j]:
                dp[i+1][j+1] = dp[i][j] + 1
            else:
                dp[i+1][j+1] = max(dp[i][j+1], dp[i+1][j])
    lcs = dp[m][n]
    prec = lcs / max(n, 1)
    rec = lcs / max(m, 1)
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def _get_first(batch: dict, keys, default=None):
    for k in keys:
        if k in batch and batch[k] is not None:
            return batch[k]
    return default


def _resolve_amp_dtype(amp_dtype):
    # Accept torch.dtype directly
    if isinstance(amp_dtype, torch.dtype):
        return amp_dtype

    # Accept strings
    if isinstance(amp_dtype, str):
        s = amp_dtype.strip().lower()
        if s in ("fp16", "float16", "half", "16"):
            return torch.float16
        if s in ("bf16", "bfloat16"):
            return torch.bfloat16
        if s in ("fp32", "float32", "32"):
            return torch.float32  # usually not useful for autocast, but safe
    # Fallback
    return torch.float16


@torch.no_grad()
def evaluate_multiclass(
    model,
    val_loader,
    device="cuda",
    amp=True,
    amp_dtype=torch.float16,
    criterion=None,
    num_classes=7,
    labels=None,
    label_names=None,   # list[str] length num_classes, optional for BLEU/ROUGE
    debug_first_batch=False,
):
    """
    Expects val_loader to yield either:
      - (x, y)
      - or a dict with keys like {"image": x, "label": y, ...}
      - optionally can include text:
            y_text (reference strings)
            pred_text (predicted strings)  [rare]
        OR we can map numeric preds to text using label_names.

    Returns:
        metrics dict
    """
    model.eval()

    if labels is None:
        labels = list(range(num_classes))

    all_probs = []
    all_true = []
    all_pred = []

    # For BLEU/ROUGE
    ref_texts = []
    hyp_texts = []

    total_loss = 0.0
    total_n = 0

    for i, batch in enumerate(val_loader):
        # --- unpack ---
        if isinstance(batch, (tuple, list)):
            x, y = batch[0], batch[1]
            extra = batch[2:] if len(batch) > 2 else None
        elif isinstance(batch, dict):
            x = _get_first(batch, ["image", "images", "x", "pixel_values"])
            y = _get_first(batch, ["label", "labels", "y", "target", "targets"])
            extra = batch
#        elif isinstance(batch, dict):
#            # adapt these keys to your dataset if needed
#            #x = batch.get("image", None) or batch.get("x", None) or batch.get("images", None)
#            x = None
#            for k in ("image", "x", "images"):
#                if k in batch and batch[k] is not None:
#                    x = batch[k]
#                    break
#            if x is None:
#                raise KeyError(f"No image tensor found in batch keys: {list(batch.keys())}")
#
#            y = batch.get("label", None) or batch.get("y", None) or batch.get("labels", None)
#            extra = batch
        else:
            raise TypeError(f"Unsupported batch type: {type(batch)}")

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True).long().view(-1)

        
        amp_dtype = _resolve_amp_dtype(amp_dtype)
        # --- forward ---
        if amp:
            with torch.autocast(device_type="cuda", dtype=amp_dtype):
                out = model(x)
        else:
            out = model(x)

        logits = out[0] if isinstance(out, (tuple, list)) else out

        if debug_first_batch and i == 0:
            print("logits:", tuple(logits.shape), logits.dtype, "y:", tuple(y.shape), y.dtype, flush=True)

        # Ensure logits shape is [B, C]
        if logits.dim() > 2:
            # If your model returns [B, T, C] or similar, reduce it.
            # Prefer mean over token dimension (T). Adjust if you need CLS token, etc.
            logits = logits.mean(dim=1)

        # loss
        if criterion is not None:
            loss = criterion(logits.float(), y)
            total_loss += float(loss.item()) * y.size(0)

        probs = torch.softmax(logits.float(), dim=-1)
        pred = torch.argmax(probs, dim=-1)

        all_probs.append(probs.detach().cpu().numpy())
        all_true.append(y.detach().cpu().numpy())
        all_pred.append(pred.detach().cpu().numpy())

        total_n += y.size(0)

        # --- Text metrics collection (optional) ---
        # 1) If batch provides y_text/pred_text
        if isinstance(extra, dict):
            y_text = extra.get("y_text", None) or extra.get("text", None) or extra.get("label_text", None)
            pred_text = extra.get("pred_text", None)

            if y_text is not None:
                # y_text can be list[str] for batch
                if isinstance(y_text, (list, tuple)):
                    ref_texts.extend([str(t) for t in y_text])
                else:
                    # single string replicated if needed
                    ref_texts.extend([str(y_text)] * int(y.size(0)))

                # if pred_text exists, use it; else map pred ids -> label_names if possible
                if pred_text is not None:
                    if isinstance(pred_text, (list, tuple)):
                        hyp_texts.extend([str(t) for t in pred_text])
                    else:
                        hyp_texts.extend([str(pred_text)] * int(y.size(0)))
                elif label_names is not None:
                    hyp_texts.extend([label_names[int(p)] for p in pred.detach().cpu().numpy().tolist()])

        # 2) If no text in batch but label_names is provided, we can compute BLEU/ROUGE on label strings
        #    by mapping y and pred to label strings. This is "label-text BLEU/ROUGE" (not generation).
        if (label_names is not None) and (len(ref_texts) < sum(len(a) for a in all_true)):  # crude guard
            # Only add if we haven't already added text from batch
            pass

    probs = np.concatenate(all_probs, axis=0)
    true = np.concatenate(all_true, axis=0)
    pred = np.concatenate(all_pred, axis=0)

    # ---- specificity/sensitivity via confusion matrix ----
    cm = confusion_matrix(true, pred, labels=labels)
    specs = []
    for k in range(num_classes):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        tn = cm.sum() - (tp + fp + fn)
        denom = (tn + fp)
        specs.append((tn / denom) if denom > 0 else 0.0)

    specificity_macro = float(np.mean(specs))
    sensitivity_macro = float(recall_score(true, pred, average="macro", zero_division=0))

    metrics = {
        "acc": float(accuracy_score(true, pred)),
        "bacc": float(balanced_accuracy_score(true, pred)),
        "f1_macro": float(f1_score(true, pred, average="macro")),
        "f1_weighted": float(f1_score(true, pred, average="weighted")),
        "precision_macro": float(precision_score(true, pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(true, pred, average="macro", zero_division=0)),
        "precision_weighted": float(precision_score(true, pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(true, pred, average="weighted", zero_division=0)),
        "sensitivity_macro": sensitivity_macro,
        "specificity_macro": specificity_macro,
        "log_loss": float(log_loss(true, probs, labels=labels)),
    }

    if criterion is not None:
        metrics["val_loss"] = float(total_loss / max(total_n, 1))

    # ---- BLEU / ROUGE-L (optional) ----
    # If we didn't get explicit texts, but you pass label_names, we can map labels to strings.
    if label_names is not None and len(ref_texts) == 0:
        ref_texts = [label_names[int(t)] for t in true.tolist()]
        hyp_texts = [label_names[int(p)] for p in pred.tolist()]

    if len(ref_texts) > 0 and len(hyp_texts) == len(ref_texts):
        # ROUGE-L
        rouge_ls = [_rouge_l_f1(h, r) for h, r in zip(hyp_texts, ref_texts)]
        metrics["rougeL_f1"] = float(np.mean(rouge_ls))

        # BLEU (sentence-level)
        if _HAS_NLTK:
            smoothie = SmoothingFunction().method1
            bleu_scores = []
            for h, r in zip(hyp_texts, ref_texts):
                ref_tokens = r.split()
                hyp_tokens = h.split()
                if len(hyp_tokens) == 0 or len(ref_tokens) == 0:
                    bleu_scores.append(0.0)
                else:
                    bleu_scores.append(float(sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)))
            metrics["bleu"] = float(np.mean(bleu_scores))
        else:
            metrics["bleu"] = None  # tell you NLTK isn't installed

    return metrics

#@torch.no_grad()
#def evaluate_multiclass(model, val_loader, device="cuda", amp=True, criterion=None, amp_dtype="fp16"):
#    model_was_training = model.training
#    model.eval()
#    if amp:
#        dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
#    
#    use_amp = bool(amp) and str(device).startswith("cuda") and torch.cuda.is_available()
#    dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
#    total_loss = 0.0
#    n_samples = 0
#    
#    all_probs = []
#    all_pred = []
#    all_true = []
#    losses = []
#
#    for batch in tqdm(val_loader, desc="eval", leave=False):
#        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
#            x, y = batch[0], batch[1]
#        else:
#            # common alternative:
#            x, y = batch["image"], batch["label"]
#        x = batch["image"].to(device, non_blocking=True)
#        y = batch["label"].to(device, non_blocking=True)
#        
#        loss = None
#        #dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
#        with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
#        # logits: [B, C]
#            # Ensure logits are [B, C]
#            # DEBUG (keep for now)
#            #print("logits:", tuple(logits.shape), logits.dtype, "y:", tuple(y.shape), y.dtype)
#            
#            # Make sure logits are [B, C] for CrossEntropyLoss
#            # --- force CE compatible targets ---
#            if logits.ndim != 2:
#                raise RuntimeError(f"logits must be [B,C], got {tuple(logits.shape)}")
#            B, C = logits.shape
#            
#            y = y.to(logits.device)
#            
#            # If you accidentally built one-hot targets earlier: [B,C] -> [B]
#            if y.ndim == 2 and y.shape == (B, C):
#                y = y.argmax(dim=1)
#            if logits.ndim > 2:
#                C = logits.size(-1)
#                # case: [B, T, C]  (very common)
#                if C == num_classes:
#                    logits = logits.mean(dim=1)
#                # case: [B, C, T]
#                elif logits.size(1) == num_classes:
#                    logits = logits.mean(dim=-1)
#                else:
#                    # fallback: flatten all non-batch dims except class if possible
#                    raise ValueError(f"Unexpected logits shape for CE: {logits.shape} (num_classes={num_classes})")
#            
#            # Targets must be [B] long
#            if y.ndim != 1:
#                y = y.view(-1)
#            y = y.long()
#            print(f"[DBG] logits shape={tuple(logits.shape)} dtype={logits.dtype} "
#            f"| y shape={tuple(y.shape)} dtype={y.dtype} device={y.device}", flush=True)
#
#            loss = criterion(logits, y)
#
#
#        #with torch.cuda.amp.autocast(enabled=amp):
##            logits = model(x)
##            if criterion is not None:
##                #losses.append(float(criterion(logits, y).detach().cpu()))
##                loss = criterion(logits, y)
#                
#        bs = x.size(0)
#        if loss is not None:
#            total_loss += float(loss.item()) * bs
#            n_samples += bs
#
#        probs = torch.softmax(logits.float(), dim=-1)
#        pred = probs.argmax(dim=-1)
#
#        all_probs.append(_to_numpy(probs))
#        all_pred.append(_to_numpy(pred))
#        all_true.append(_to_numpy(y))
#    probs = np.concatenate(all_probs, axis=0)
#    pred  = np.concatenate(all_pred, axis=0)
#    true  = np.concatenate(all_true, axis=0)
#    
#    num_classes = probs.shape[1]
#    labels = list(range(num_classes))
#    
#     
#    
#    # ------------------------------
#    # Optional text metrics (BLEU / ROUGE-L / BERTScore)
#    # Only computed if you have both pred_texts and ref_texts available.
#    # If not present, this section does nothing.
#    # ------------------------------
#    pred_texts = locals().get("pred_texts", None)
#    ref_texts  = locals().get("ref_texts", None)
#    
#    if pred_texts is not None and ref_texts is not None:
#        try:
#            pred_texts = [str(x) for x in pred_texts]
#            ref_texts  = [str(x) for x in ref_texts]
#            assert len(pred_texts) == len(ref_texts)
#        except Exception:
#            pred_texts, ref_texts = None, None
#    
#    if pred_texts is not None and ref_texts is not None and evaluate is not None:
#        # BLEU (sacrebleu)
#        try:
#            bleu = evaluate.load("sacrebleu")
#            metrics["bleu"] = float(bleu.compute(
#                predictions=pred_texts,
#                references=[[r] for r in ref_texts]
#            )["score"])
#        except Exception:
#            pass
#    
#        # ROUGE-L
#        try:
#            rouge = evaluate.load("rouge")
#            r = rouge.compute(predictions=pred_texts, references=ref_texts, use_stemmer=True)
#            metrics["rougeL"] = float(r["rougeL"])
#        except Exception:
#            pass
#    
#    if pred_texts is not None and ref_texts is not None and bertscore is not None:
#        try:
#            P, R, F1 = bertscore(pred_texts, ref_texts, lang="en", rescale_with_baseline=True)
#            metrics["bertscore_f1"] = float(F1.mean().item())
#        except Exception:
#            pass
#
#
##    probs = np.concatenate(all_probs, axis=0)
##    pred = np.concatenate(all_pred, axis=0)
##    true = np.concatenate(all_true, axis=0)
##
##    metrics = {
##        "acc": float(accuracy_score(true, pred)),
##        "bacc": float(balanced_accuracy_score(true, pred)),
##        "f1_macro": float(f1_score(true, pred, average="macro")),
##        "f1_weighted": float(f1_score(true, pred, average="weighted")),
##        "precision_macro": float(precision_score(true, pred, average="macro", zero_division=0)),
##        "recall_macro": float(recall_score(true, pred, average="macro", zero_division=0)),
##        "log_loss": float(log_loss(true, probs, labels=list(range(probs.shape[1])))),
##    }
#
#    # AUROC OVR macro (guard for small class counts)
#    try:
#        metrics["auroc_ovr_macro"] = float(roc_auc_score(true, probs, multi_class="ovr", average="macro"))
#    except Exception:
#        metrics["auroc_ovr_macro"] = float("nan")
#
#    if losses:
#        metrics["loss"] = float(np.mean(losses))
#        
#    # Restore mode
#    if model_was_training:
#        model.train()
#
#    return metrics


# ------------------------
# Training
# ------------------------
def train_multiclass(model, train_loader, val_loader, cfg, run_dir, wandb=None):
    device = torch.device(cfg.get("device", "cuda"))
    model.to(device)

    train_cfg = cfg.get("train", {}) or {}
    eval_cfg = cfg.get("eval", {}) or {}

    epochs = int(train_cfg.get("epochs", 10))
    lr = float(train_cfg.get("lr", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.0))
    amp = bool(train_cfg.get("amp", True))
    log_every = int(train_cfg.get("log_every", 50))
    grad_clip = float(train_cfg.get("grad_clip", 0.0) or 0.0)
    accum_steps = int(train_cfg.get("accum_steps", 1) or 1)

    # selection for best checkpoint
    select_metric = str(eval_cfg.get("select_metric", "f1_macro"))
    select_mode = str(eval_cfg.get("select_mode", "max")).lower()  # max|min
    report_metrics = eval_cfg.get("report_metrics", None)

    os.makedirs(run_dir, exist_ok=True)
    metrics_path = os.path.join(run_dir, "metrics.jsonl")
    
    bad = [(n, p.dtype, p.shape) for n,p in model.named_parameters()
       if p.requires_grad and p.dtype == torch.float16]

    print("FP16 trainable params:", len(bad))
    for x in bad[:40]:
        print(x)
        
    # after requires_grad flags are set
    if hasattr(model, "backbone") and hasattr(model.backbone, "visual_encoder"):
        model.backbone.visual_encoder.float()
    
    # also often needed
    if hasattr(model.backbone, "ln_vision") and model.backbone.ln_vision is not None:
        model.backbone.ln_vision.float()
    
    # if you also train Qformer
    if hasattr(model.backbone, "Qformer") and model.backbone.Qformer is not None:
        model.backbone.Qformer.float()
    
    # your classification head should also be fp32
    if hasattr(model, "classifier"):
        model.classifier.float()


    # Optimizer over trainable parameters only
    params = [p for p in model.parameters() if p.requires_grad]
    opt_cfg = train_cfg.get("optimizer", {}) or {}
    opt_name = str(opt_cfg.get("name", "adamw")).lower()
    betas = tuple(opt_cfg.get("betas", [0.9, 0.999]))
    eps = float(opt_cfg.get("eps", 1e-8))

    if opt_name in ("adamw", "adam"):
        opt = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay, betas=betas, eps=eps)
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name}")

    # Scheduler (cosine + warmup)
    sched_cfg = train_cfg.get("scheduler", {}) or {}
    sched_name = str(sched_cfg.get("name", "none")).lower()
    total_steps = max(1, epochs * max(1, len(train_loader)) // accum_steps)
    warmup_steps = int(sched_cfg.get("warmup_steps", 0) or 0)
    warmup_ratio = float(sched_cfg.get("warmup_ratio", 0.0) or 0.0)
    if warmup_steps <= 0 and warmup_ratio > 0:
        warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = max(0, min(warmup_steps, total_steps - 1))
    min_lr = float(sched_cfg.get("min_lr", 0.0) or 0.0)

    scheduler = None
    if sched_name in ("cosine", "cosine_warmup", "cosine_with_warmup"):
        def lr_lambda(step: int):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            # cosine decay
            progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
            cosine = 0.5 * (1.0 + np.cos(np.pi * progress))
            min_factor = min_lr / lr if lr > 0 else 0.0
            return min_factor + (1.0 - min_factor) * cosine

        scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    elif sched_name in ("none", "", None):
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {sched_name}")

    #scaler = torch.cuda.amp.GradScaler(enabled=amp)
    use_amp = True
    #use_amp = bool(amp) and (device.startswith("cuda") or device == "cuda")
    amp_dtype = "bf16"  # or read from cfg
    
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and amp_dtype == "fp16"))

    # loss
    num_classes = int(cfg.get("num_classes", train_cfg.get("num_classes", 7)))
    criterion = build_loss(cfg, num_classes=num_classes, device=device)
    print("LOSS OBJECT:", criterion)
    print("LOSS CLASS :", criterion.__class__)

    best_score = None
    best_path = os.path.join(run_dir, "best_trainable.pt")
    last_path = os.path.join(run_dir, "last_trainable.pt")

    global_step = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running = []

        opt.zero_grad(set_to_none=True)

        for it, batch in enumerate(tqdm(train_loader, desc=f"train e{epoch}", leave=False), start=1):
            x = batch["image"].to(device, non_blocking=True)
            y = batch["label"].to(device, non_blocking=True)

            use_amp = bool(amp)
            dtype = torch.float16 if amp_dtype == "fp16" else torch.bfloat16
            with torch.amp.autocast("cuda", dtype=dtype, enabled=use_amp):
            #with torch.cuda.amp.autocast(enabled=amp):
                logits = model(x)
                loss = criterion(logits.float(), y)
                loss = loss / float(accum_steps)

            scaler.scale(loss).backward()

            if (it % accum_steps) == 0:
                if grad_clip and grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, grad_clip)

                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)

                if scheduler is not None:
                    scheduler.step()

                global_step += 1

                running.append(float(loss.detach().cpu()) * float(accum_steps))

                if log_every and (global_step % log_every == 0):
                    lr_now = opt.param_groups[0]["lr"]
                    print(f"[step {global_step}] train/loss={np.mean(running[-log_every:]):.4f} lr={lr_now:.3e}")
                    if wandb:
                        wandb.log(
                            {
                                "step": global_step,
                                "train/loss": float(np.mean(running[-log_every:])),
                                "lr": float(lr_now),
                            },
                            step=global_step,
                        )

        train_loss = float(np.mean(running)) if running else float("nan")

        # Validation (per-epoch)
        #val_metrics = evaluate_multiclass(model, val_loader, device=device, amp=amp, criterion=criterion)
        amp_dtype = getattr(cfg, "amp_dtype", "fp16")  # or read from your sub_cfg dict
        val_metrics = evaluate_multiclass(model, val_loader, device=device, amp=amp, criterion=criterion, amp_dtype=amp_dtype)
        val_loss = float(val_metrics.get("loss", float("nan")))

        # choose best score
        score = float(val_metrics.get(select_metric, val_loss))
        is_best = False
        if best_score is None:
            is_best = True
        else:
            if select_mode == "min":
                is_best = score < best_score
            else:
                is_best = score > best_score

        if is_best:
            best_score = score
            torch.save({"trainable": get_trainable_state_dict(model), "epoch": epoch, "score": score}, best_path)

        # Always save last
        torch.save({"trainable": get_trainable_state_dict(model), "epoch": epoch, "score": score}, last_path)

        # Log metrics row
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "lr": float(opt.param_groups[0]["lr"]),
            "train_loss": train_loss,
            "val_loss": val_loss,
            "best": bool(is_best),
            "select_metric": select_metric,
            "score": float(score),
            "val_metrics": {k: float(v) for k, v in val_metrics.items()},
        }

        with open(metrics_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

        # write a compact "latest" file for quick parsing
        safe_write_json(os.path.join(run_dir, "last_metrics.json"), row)

        # Console summary (filter report_metrics if provided)
        if isinstance(report_metrics, (list, tuple)) and len(report_metrics) > 0:
            rep = {k: float(val_metrics.get(k, float('nan'))) for k in report_metrics}
        else:
            rep = {k: float(v) for k, v in val_metrics.items()}

        rep_str = " ".join([f"{k}={v:.4f}" for k, v in rep.items() if isinstance(v, (int, float)) and not np.isnan(v)])
        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} val_loss={val_loss:.4f} {select_metric}={score:.4f} " +
            ("(BEST) " if is_best else "") + rep_str
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
