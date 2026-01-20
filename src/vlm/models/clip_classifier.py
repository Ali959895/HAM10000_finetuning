# -*- coding: utf-8 -*-
"""CLIP classifier wrapper (OpenCLIP preferred).

This wrapper is intentionally simple:
- Uses OpenCLIP if available (recommended for ViT-L/14, RN50, etc.)
- Freezes backbone by default; you can unfreeze last N blocks for ViT backbones
- Adds a configurable MLP head for classification
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from lavis.models import load_model_and_preprocess
import open_clip
def resize_attnpool_pos_embed(visual, image_size: int):
    """
    Resize open_clip ModifiedResNet AttentionPool2d positional embedding
    so the model can run at a different image_size than it was pretrained with.
    """
    # ViT models: nothing to do (they interpolate grid pos-embed internally if needed)
    if visual.__class__.__name__.lower().find("visiontransformer") >= 0:
        return

    # ResNet models: open_clip uses either 'attnpool' or 'attn_pool' depending on version
    attnpool = getattr(visual, "attnpool", None) or getattr(visual, "attn_pool", None)
    if attnpool is None:
        return
    
    attnpool = visual.attnpool
    pos = attnpool.positional_embedding  # shape: (HW+1, C)

    # Compute feature map size after ResNet stem + layers in CLIP modified resnet:
    # For CLIP ResNet, output stride is 32 => grid = image_size // 32
    grid = image_size // 32
    new_hw = grid * grid
    c = pos.shape[1]

    old_hw1 = pos.shape[0] - 1
    old_grid = int(old_hw1 ** 0.5)
    if old_grid * old_grid != old_hw1:
        raise ValueError(f"Unexpected old pos_embed length: {pos.shape[0]}")

    if old_grid == grid:
        return  # nothing to do

    # Separate cls token and grid tokens
    cls_pos = pos[:1, :]                      # (1, C)
    grid_pos = pos[1:, :].reshape(old_grid, old_grid, c).permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    # Interpolate
    grid_pos = F.interpolate(grid_pos, size=(grid, grid), mode="bicubic", align_corners=False)

    # Repack
    grid_pos = grid_pos.squeeze(0).permute(1, 2, 0).reshape(new_hw, c)  # (HW, C)
    new_pos = torch.cat([cls_pos, grid_pos], dim=0)                     # (HW+1, C)

    attnpool.positional_embedding = torch.nn.Parameter(new_pos)


def _get_activation(name: str) -> nn.Module:
    name = (name or "gelu").lower()
    if name in ("relu",):
        return nn.ReLU(inplace=True)
    if name in ("gelu",):
        return nn.GELU()
    if name in ("silu", "swish"):
        return nn.SiLU()
    if name in ("tanh",):
        return nn.Tanh()
    if name in ("none", "identity", ""):
        return nn.Identity()
    raise ValueError(f"Unknown activation: {name}")


def _freeze_all(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def _unfreeze_last_openclip_vit_blocks(visual: nn.Module, last_n: int) -> None:
    """Unfreeze last N transformer blocks for OpenCLIP ViT.

    OpenCLIP ViT usually stores blocks at: visual.transformer.resblocks (ModuleList).
    """
    if last_n <= 0:
        return
    transformer = getattr(visual, "transformer", None)
    resblocks = getattr(transformer, "resblocks", None) if transformer is not None else None
    if resblocks is None:
        return
    n = len(resblocks)
    last_n = min(int(last_n), n)
    for blk in list(resblocks)[n - last_n :]:
        for p in blk.parameters():
            p.requires_grad = True

    # also unfreeze post-norm / ln_post if present
    for attr in ("ln_post", "norm", "fc_norm"):
        ln = getattr(visual, attr, None)
        if ln is not None:
            for p in ln.parameters():
                p.requires_grad = True


class ClipClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int,
        device: str = "cuda",
        model_name: str = "ViT-L-14",
        pretrained: str = "openai",
        image_size: int = 224,   
        # training knobs
        train_vision: bool = False,
        unfreeze_vision_last_n: int = 0,
        # head knobs
        pooling: str = "cls",
        head_hidden: int = 0,
        activation: str = "gelu",
        dropout: float = 0.0,
        normalize_features: bool = True,
    ):
        super().__init__()
        self.use_openclip = True
        self.num_classes = int(num_classes)
        self.device_str = device
        self.model_name = model_name
        self.pretrained = pretrained
        self.pooling = pooling
        self.normalize_features = bool(normalize_features)

        # Prefer OpenCLIP
        # get desired image size from config, default to 224
        image_size = int(image_size)

        try:
            self.backbone, self.preprocess_train, self.preprocess_val = open_clip.create_model_and_transforms(
                model_name,
                pretrained=pretrained,
            )
            visual = self.backbone.visual
            if hasattr(visual, "attnpool"):
                resize_attnpool_pos_embed(visual, image_size=int(image_size))

        except Exception as e:
            self.use_openclip = False
            raise RuntimeError( f"OpenCLIP init failed for model={model_name} pretrained={pretrained}. "
            f"Not falling back to LAVIS (would require internet). Original error: {e}")

            self.use_openclip = False

        _freeze_all(self.backbone)

        # Unfreeze vision if requested
        if train_vision or (unfreeze_vision_last_n and int(unfreeze_vision_last_n) > 0):
            visual = getattr(self.backbone, "visual", None) or getattr(self.backbone, "visual_encoder", None)
            if visual is not None:
                if unfreeze_vision_last_n and int(unfreeze_vision_last_n) > 0:
                    _unfreeze_last_openclip_vit_blocks(visual, int(unfreeze_vision_last_n))
                else:
                    for p in visual.parameters():
                        p.requires_grad = True

        feat_dim = self._infer_feat_dim()
        act = _get_activation(activation)
        if head_hidden and int(head_hidden) > 0:
            self.classifier = nn.Sequential(
                nn.Linear(feat_dim, int(head_hidden)),
                act,
                nn.Dropout(float(dropout)),
                nn.Linear(int(head_hidden), self.num_classes),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(float(dropout)),
                nn.Linear(feat_dim, self.num_classes),
            )

    @torch.no_grad()
    def _infer_feat_dim(self) -> int:
        # OpenCLIP models commonly expose embed_dim
        if hasattr(self.backbone, "embed_dim"):
            return int(getattr(self.backbone, "embed_dim"))
        # OpenCLIP vision often has output_dim
        visual = getattr(self.backbone, "visual", None)
        if visual is not None and hasattr(visual, "output_dim"):
            return int(getattr(visual, "output_dim"))

        # fallback: run a tiny forward
        dev = torch.device(self.device_str)
        x = torch.zeros(1, 3, 224, 224, device=dev)
        try:
            f = self._encode_image(x)
            return int(f.shape[-1])
        except Exception:
            return 768

    def _encode_image(self, images: torch.Tensor) -> torch.Tensor:
        if self.use_openclip:
            f = self.backbone.encode_image(images)
        else:
            # LAVIS clip model may expose encode_image too; if not, try forward
            if hasattr(self.backbone, "encode_image"):
                f = self.backbone.encode_image(images)
            else:
                f = self.backbone(images)
        return f

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        f = self._encode_image(images)
        # f is [B, D]
        if f.dim() > 2:
            # some backbones may return token sequences; pool simply
            f = f[:, 0] if self.pooling == "cls" else f.mean(dim=1)

        if self.normalize_features:
            f = f / (f.norm(dim=-1, keepdim=True) + 1e-6)

        return self.classifier(f)
