# src/lavis_ext/models/clip_vitl_multilabel.py

import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip
from safetensors.torch import load_file

from lavis.common.registry import registry
from lavis.models.base_model import BaseModel


@registry.register_model("clip_vitl_multilabel")
class CLIPViTLMultiLabel(BaseModel):
    """
    LAVIS model wrapper: OpenCLIP ViT-L/14 image tower + linear head for COCO multi-label.
    """

    # Important for LAVIS Config() when it tries to resolve model_type -> config path
    PRETRAINED_MODEL_CONFIG_DICT = {
        "clip_vitl_multilabel": "configs/models/clip_vitl_multilabel.yaml"
    }

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

        # Build OpenCLIP model offline
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name=model_name,
            pretrained=None,
            device="cpu",
        )

        # Load local pretrained safetensors (offline)
        if pretrained_path:
            state = load_file(pretrained_path)
            missing, unexpected = self.clip_model.load_state_dict(state, strict=False)
            print(f"[INFO] Loaded pretrained weights from: {pretrained_path}", flush=True)
            print(f"[INFO] Missing keys: {len(missing)} | Unexpected keys: {len(unexpected)}", flush=True)

        # Freeze text tower (prevents DDP unused-parameter crash since we only use encode_image)
        # ---- Freeze text tower (important for image-only classification) ----
        if freeze_text:
            for name, p in self.clip_model.named_parameters():
                # for classification we only use image encoder, so freeze everything not in visual tower
                if not name.startswith("visual"):
                    p.requires_grad_(False)
        
        # logit_scale/logit_bias are not used in BCE classification loss
        ls = getattr(self.clip_model, "logit_scale", None)
        if ls is not None and hasattr(ls, "requires_grad_"):
            ls.requires_grad_(False)
        
        lb = getattr(self.clip_model, "logit_bias", None)
        if lb is not None and hasattr(lb, "requires_grad_"):
            lb.requires_grad_(False)

        # Classifier head
        embed_dim = self.clip_model.visual.output_dim
        self.classifier = nn.Linear(embed_dim, num_classes)

    @classmethod
    def default_config_path(cls, model_type="clip_vitl_multilabel"):
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
            out["loss"] = F.binary_cross_entropy_with_logits(logits, labels)

        return out
