from __future__ import annotations
from typing import Dict
import torch
from .metrics import compute_AUC, pfbeta_binarized

@torch.no_grad()
def multilabel_metrics_from_logits(y_true: torch.Tensor, logits: torch.Tensor) -> Dict[str, float]:
    y_prob = torch.sigmoid(logits)
    auroc, auprc = compute_AUC(y_true, y_prob)
    try:
        from sklearn.metrics import average_precision_score
        mAP = float(average_precision_score(y_true.cpu().numpy(), y_prob.cpu().numpy()))
    except Exception:
        mAP = float("nan")
    return {"AUROC": float(auroc), "AUPRC": float(auprc), "mAP": mAP}

@torch.no_grad()
def binary_metrics_from_logits(y_true: torch.Tensor, logits: torch.Tensor) -> Dict[str, float]:
    y_prob = torch.sigmoid(logits).view(-1).cpu().numpy()
    y_true_np = y_true.view(-1).cpu().numpy().astype(int)
    try:
        from sklearn.metrics import roc_auc_score, average_precision_score
        auroc = float(roc_auc_score(y_true_np, y_prob))
        auprc = float(average_precision_score(y_true_np, y_prob))
    except Exception:
        auroc, auprc = float("nan"), float("nan")
    try:
        pfb = float(pfbeta_binarized(y_true_np, y_prob))
    except Exception:
        pfb = float("nan")
    return {"AUROC": auroc, "AUPRC": auprc, "pfbeta": pfb}
