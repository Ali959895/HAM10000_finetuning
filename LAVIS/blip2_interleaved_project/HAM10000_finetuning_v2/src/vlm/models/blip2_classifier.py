# -*- coding: utf-8 -*-
"""
BLIP-2 / CLIP vision backbones for HAM10000 classification.

Supports:
- LAVIS "blip2_opt" with model_type "pretrain_opt2.7b" (EVA ViT-g/14 + Q-Former + OPT)
- LAVIS "blip2_feature_extractor" with model_type "pretrain_vitL" (CLIP ViT-L/14 + Q-Former, no OPT)

Key knobs:
- train_qformer: fine-tune Q-Former + projection (and OPT if you ever enable it)
- train_vision + unfreeze_vision_last_n: fine-tune last N ViT blocks (1–6 typical)
- head_hidden / activation / dropout: classification head
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
import torch.nn as nn


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


def _unfreeze_last_vit_blocks(vit: nn.Module, last_n: int) -> None:
    """
    Unfreeze last N transformer blocks of a ViT-like module.

    Works with timm ViT-style modules that expose `blocks` as a list/ModuleList.
    """
    if last_n <= 0:
        return
    blocks = getattr(vit, "blocks", None)
    if blocks is None:
        # Fallback: unfreeze everything (better than silently doing nothing)
        for p in vit.parameters():
            p.requires_grad = True
        return
    # Freeze everything first
    for p in vit.parameters():
        p.requires_grad = False
    # Unfreeze last N blocks
    n = len(blocks)
    for blk in blocks[max(0, n - last_n):]:
        for p in blk.parameters():
            p.requires_grad = True
    # Also unfreeze final norm if exists (often important)
    for attr in ("norm", "fc_norm", "ln_post"):
        layer = getattr(vit, attr, None)
        if layer is not None:
            for p in layer.parameters():
                p.requires_grad = True


class Blip2Classifier(nn.Module):
    """
    A unified classifier that can run on:
      - LAVIS blip2_opt (pretrain_opt2.7b) -> use Q-Former query features projected to OPT hidden
      - LAVIS blip2_feature_extractor (pretrain_vitL / pretrain) -> use extract_features("image")
    """

    def __init__(
        self,
        num_classes: int,
        lavis_name: str = "blip2_opt",
        model_type: str = "pretrain_opt2.7b",
        device: str = "cuda",
        # training knobs
        train_qformer: bool = False,
        train_vision: bool = False,
        unfreeze_vision_last_n: int = 0,
        # representation / head
        pooling: str = "mean",  # "mean" or "cls" (if supported)
        head_hidden: int = 0,   # 0 -> linear head
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.lavis_name = lavis_name
        self.model_type = model_type
        self.device_str = device
        self.pooling = pooling

        # Import lazily so this file can be imported outside LAVIS env.
        from lavis.models import load_model_and_preprocess

        model, _, _ = load_model_and_preprocess(
            name=lavis_name, model_type=model_type, is_eval=False, device=device
        )
        self.backbone = model
        self.backbone_device = device

        # ===== Freeze policy =====
        # Always freeze everything first, then selectively unfreeze.
        _freeze_all(self.backbone)

        # Vision encoder handling
        self.visual = getattr(self.backbone, "visual_encoder", None) or getattr(self.backbone, "visual_encoder", None)
        if self.visual is None:
            self.visual = getattr(self.backbone, "visual_encoder", None)  # keep for safety

        if train_vision or unfreeze_vision_last_n > 0:
            # If last_n is specified, unfreeze only last N blocks.
            # Otherwise unfreeze all vision parameters.
            if self.visual is not None:
                if unfreeze_vision_last_n > 0:
                    _unfreeze_last_vit_blocks(self.visual, int(unfreeze_vision_last_n))
                else:
                    for p in self.visual.parameters():
                        p.requires_grad = True

            # Also unfreeze ln_vision (used in blip2_opt)
            ln_vision = getattr(self.backbone, "ln_vision", None)
            if ln_vision is not None:
                for p in ln_vision.parameters():
                    p.requires_grad = True

        # Q-Former / projection handling (blip2_opt and blip2_feature_extractor both have Qformer)
        if train_qformer:
            for attr in ("Qformer", "query_tokens", "opt_proj", "text_proj", "vision_proj"):
                m = getattr(self.backbone, attr, None)
                if m is None:
                    continue
                if isinstance(m, torch.Tensor):
                    m.requires_grad = True
                else:
                    for p in m.parameters():
                        p.requires_grad = True

        # ===== Determine feature dim and create head =====
        feat_dim = self._infer_feature_dim()
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

    def _infer_feature_dim(self) -> int:
        # For blip2_opt, use opt_proj output dim if present.
        opt_proj = getattr(self.backbone, "opt_proj", None)
        if opt_proj is not None and hasattr(opt_proj, "out_features"):
            return int(opt_proj.out_features)

        # For feature extractor, try known projection heads.
        for attr in ("vision_proj", "text_proj"):
            m = getattr(self.backbone, attr, None)
            if m is not None and hasattr(m, "out_features"):
                return int(m.out_features)

        # Fallback: run a tiny forward with a dummy tensor is too costly / risky.
        # Default to 768.
        return 768

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns logits [B, num_classes]
        """
        feats = self.encode_image(images)
        return self.classifier(feats)
    def encode_image(self, images: torch.Tensor) -> torch.Tensor:
        """
        Returns a pooled feature [B, D] on the same device as the backbone.
        """
        device = next(self.backbone.parameters()).device
        images = images.to(device, non_blocking=True)

        if hasattr(self.backbone, "extract_features"):
            # blip2_feature_extractor path
            out = self.backbone.extract_features({"image": images}, mode="image")
            # LAVIS outputs may expose image_embeds or image_embeds_proj
            img_embeds = getattr(out, "image_embeds", None)
            if img_embeds is None:
                img_embeds = out.get("image_embeds", None) if isinstance(out, dict) else None
            if img_embeds is None:
                raise RuntimeError("extract_features did not return image_embeds")
            # img_embeds: [B, T, D]
            if self.pooling == "cls":
                pooled = img_embeds[:, 0]
            else:
                pooled = img_embeds.mean(dim=1)
            return pooled

        # blip2_opt path (no extract_features): use visual encoder + Q-Former query output projected to OPT dim
        if not hasattr(self.backbone, "Qformer"):
            raise RuntimeError(f"Backbone {type(self.backbone)} does not support image encoding")

        image_embeds = self.backbone.ln_vision(self.backbone.visual_encoder(images))
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=device)
        query_tokens = self.backbone.query_tokens.expand(image_embeds.shape[0], -1, -1)

        query_output = self.backbone.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        q = query_output.last_hidden_state[:, : query_tokens.size(1), :]  # [B, Q, H]
        if self.pooling == "cls":
            pooled = q[:, 0]
        else:
            pooled = q.mean(dim=1)
        # Project to OPT hidden size if available
        opt_proj = getattr(self.backbone, "opt_proj", None)
        if opt_proj is not None:
            pooled = opt_proj(pooled)
        return pooled


# Backward-compatible name used by older code/configs.
class Blip2MultiLabelClassifier(Blip2Classifier):
    pass
