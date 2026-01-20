import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask

def _sigmoid(x):
    return torch.sigmoid(x)

@torch.no_grad()
def multilabel_metrics(logits, targets, thresh=0.5, eps=1e-9):
    # logits: [B,C], targets: [B,C] in {0,1}
    probs = _sigmoid(logits)
    preds = (probs >= thresh).to(targets.dtype)

    tp = (preds * targets).sum(dim=0)
    fp = (preds * (1 - targets)).sum(dim=0)
    fn = ((1 - preds) * targets).sum(dim=0)

    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    f1_per_class = 2 * prec * rec / (prec + rec + eps)

    macro_f1 = f1_per_class.mean().item()

    # micro
    tp_m = tp.sum()
    fp_m = fp.sum()
    fn_m = fn.sum()
    prec_m = (tp_m / (tp_m + fp_m + eps)).item()
    rec_m  = (tp_m / (tp_m + fn_m + eps)).item()
    micro_f1 = (2 * prec_m * rec_m / (prec_m + rec_m + eps + 1e-12))

    return {
        "macro_f1@0.5": macro_f1,
        "micro_f1@0.5": micro_f1,
        "precision_micro@0.5": prec_m,
        "recall_micro@0.5": rec_m,
    }

@registry.register_task("classification_ext")
class ClassificationTask(BaseTask):
    """
    Multi-label classification (COCO 80).
    Expects samples:
      - samples["image"]: Tensor [B,3,H,W]
      - samples["label"]: Float Tensor [B,80] (0/1)
    """

    def __init__(self, **kwargs):
        super().__init__()

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        return cls()

    def train_step(self, model, samples):
        out = model(samples)
        logits = out["logits"] if isinstance(out, dict) else out
        targets = samples["label"].to(logits.device).float()
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss

    @torch.no_grad()
    def evaluation(self, model, data_loader, **kwargs):
        model.eval()
        total_loss = 0.0
        n = 0

        all_logits = []
        all_targets = []

        for samples in data_loader:
            out = model(samples)
            logits = out["logits"] if isinstance(out, dict) else out
            targets = samples["label"].to(logits.device).float()
            loss = F.binary_cross_entropy_with_logits(logits, targets)

            bs = logits.size(0)
            total_loss += loss.item() * bs
            n += bs

            all_logits.append(logits.detach().float().cpu())
            all_targets.append(targets.detach().float().cpu())

        logits_cat = torch.cat(all_logits, dim=0)
        targets_cat = torch.cat(all_targets, dim=0)

        metrics = multilabel_metrics(logits_cat, targets_cat, thresh=0.5)
        metrics["eval_loss"] = total_loss / max(n, 1)
        metrics["agg_metrics"] = metrics["macro_f1@0.5"]  # what LAVIS often logs/uses

        return metrics
