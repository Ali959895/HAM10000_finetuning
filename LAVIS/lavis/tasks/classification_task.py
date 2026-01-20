import torch
import torch.nn.functional as F
from lavis.common.registry import registry
from lavis.tasks.base_task import BaseTask


@registry.register_task("classification")
class ClassificationTask(BaseTask):
    """
    Minimal multi-label classification task for COCO-style labels.
    Expects each batch to provide:
      - samples["image"]
      - samples["label"]  (float tensor [B, 80] with 0/1)
    """

    def __init__(self, **kwargs):
        super().__init__()

    @classmethod
    def setup_task(cls, cfg, **kwargs):
        return cls()

    def train_step(self, model, samples):
        # LAVIS models usually accept dicts; keep both patterns safe
        out = model(samples)
        if isinstance(out, dict) and "loss" in out:
            return out["loss"]
        if isinstance(out, dict) and "logits" in out:
            logits = out["logits"]
        else:
            logits = out  # assume tensor

        targets = samples["label"].to(logits.device).float()
        loss = F.binary_cross_entropy_with_logits(logits, targets)
        return loss

    @torch.no_grad()
    def evaluation(self, model, data_loader, **kwargs):
        model.eval()
        losses = []
        for samples in data_loader:
            out = model(samples)
            if isinstance(out, dict) and "loss" in out:
                loss = out["loss"]
            else:
                logits = out["logits"] if isinstance(out, dict) and "logits" in out else out
                targets = samples["label"].to(logits.device).float()
                loss = F.binary_cross_entropy_with_logits(logits, targets)
            losses.append(loss.item())
        return {"eval_loss": sum(losses) / max(len(losses), 1)}
