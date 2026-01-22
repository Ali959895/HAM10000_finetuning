import torch
import torch.nn.functional as F
import torch.nn as nn

import open_clip
from safetensors.torch import load_file

from lavis.common.registry import registry
from lavis.models.base_model import BaseModel


@registry.register_model("clip_vitl_multilabel")
class CLIPViTLMultiLabel(BaseModel):
    """
    LAVIS-compatible model (must inherit BaseModel).
    OpenCLIP image encoder + linear head for multi-label classification.
    """

    def __init__(
        self,
        num_classes: int = 80,
        model_name: str = "ViT-L-14",
        pretrained_path: str | None = None,
        freeze_text: bool = True,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.model_name = model_name
        self.pretrained_path = pretrained_path
        self.freeze_text = freeze_text

        # Create OpenCLIP model (no download; we load from local file below if provided)
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=None,
            device="cpu",
        )

        # Optionally freeze text tower (prevents unused-param DDP issues + saves memory)
        if freeze_text:
            if hasattr(self.clip_model, "transformer"):
                for p in self.clip_model.transformer.parameters():
                    p.requires_grad = False
            if hasattr(self.clip_model, "token_embedding"):
                for p in self.clip_model.token_embedding.parameters():
                    p.requires_grad = False
            if hasattr(self.clip_model, "ln_final"):
                for p in self.clip_model.ln_final.parameters():
                    p.requires_grad = False
            if hasattr(self.clip_model, "text_projection"):
                tp = self.clip_model.text_projection
                if isinstance(tp, torch.nn.Parameter):
                    tp.requires_grad = False

        # Load local pretrained weights (offline)
        if pretrained_path:
            state = load_file(pretrained_path)
            missing, unexpected = self.clip_model.load_state_dict(state, strict=False)
            print(f"[INFO] Loaded pretrained weights from: {pretrained_path}", flush=True)
            print(f"[INFO] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}", flush=True)

        # Linear head
        embed_dim = self.clip_model.visual.output_dim
        self.classifier = nn.Linear(embed_dim, num_classes)

    @classmethod
    def default_config_path(cls, model_type="default"):
        # LAVIS calls this while building config (even if you override options on CLI)
        return "configs/models/clip_vitl_multilabel.yaml"
    @classmethod
    def from_config(cls, cfg):
        return cls(
            num_classes=getattr(cfg, "num_classes", 80),
            model_name=getattr(cfg, "model_name", "ViT-L-14"),
            pretrained_path=getattr(cfg, "pretrained_path", None),
            freeze_text=getattr(cfg, "freeze_text", True),
        )

    def forward(self, samples):
        images = samples["image"]
        labels = samples.get("label", None)

        feats = self.clip_model.encode_image(images)
        feats = F.normalize(feats, dim=-1)

        logits = self.classifier(feats)
        out = {"logits": logits}

        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)
            out["loss"] = loss

        return out
