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
from typing import Tuple

def _get_module_dtype_device(m):
    # Prefer parameters, then buffers
    for p in m.parameters():
        return p.dtype, p.device
    for b in m.buffers():
        return b.dtype, b.device
    return torch.float32, torch.device("cpu")


import torch

def _module_dtype_device(m, fallback_device="cuda"):
    # params first
    for p in m.parameters(recurse=True):
        return p.dtype, p.device
    # then buffers
    for b in m.buffers(recurse=True):
        return b.dtype, b.device
    # fallback
    return torch.float32, torch.device(fallback_device)


def _vision_param_dtype_device(visual: torch.nn.Module):
    """
    Return (dtype, device) that the vision encoder expects for image input.
    Works for EVA-ViT (patch_embed.proj) and CLIP-ViT (conv1).
    """
    # EVA ViT in LAVIS: visual.patch_embed.proj is Conv2d
    if hasattr(visual, "patch_embed") and hasattr(visual.patch_embed, "proj"):
        p = visual.patch_embed.proj.weight
        return p.dtype, p.device

    # CLIP ViT in LAVIS: visual.conv1 is Conv2d
    if hasattr(visual, "conv1"):
        p = visual.conv1.weight
        return p.dtype, p.device

    # fallback: first parameter
    p = next(visual.parameters())
    return p.dtype, p.device


def _cast_image_like_vision(x: torch.Tensor, visual: torch.nn.Module):
    vdtype, vdev = _vision_param_dtype_device(visual)
    if x.device != vdev or x.dtype != vdtype:
        x = x.to(device=vdev, dtype=vdtype)
    return x

#def _get_module_dtype_device(m):
#    for p in m.parameters():
#        return p.dtype, p.device
#    return torch.float32, torch.device("cpu")



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
    # IMPORTANT:
    # Many LAVIS BLIP-2 backbones are instantiated with fp16 vision weights (vit_precision="fp16").
    # If we unfreeze fp16 parameters and use GradScaler, PyTorch can raise:
    #   ValueError: Attempting to unscale FP16 gradients.
    # To keep AMP+GradScaler stable, cast *trainable* vision blocks (and final norms) to fp32.
    for blk in blocks[max(0, n - last_n):]:
        blk.float()
        for p in blk.parameters():
            p.requires_grad = True
    # Also unfreeze final norm if exists (often important)
    for attr in ("norm", "fc_norm", "ln_post"):
        layer = getattr(vit, attr, None)
        if layer is not None:
            layer.float()
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
        image_size: int = 224,
        model_cfg=None,
        # training knobs
        train_qformer: bool = False,
        train_vision: bool = False,
        unfreeze_vision_last_n: int = 0,
        # representation / head
        pooling: str = "mean",
        head_hidden: int = 0,
        activation: str = "gelu",
        dropout: float = 0.0,
    ):
        super().__init__()
    
        import torch
        import torch.nn as nn
    
        self.cfg = model_cfg or {}
        self.num_classes = int(num_classes)
        self.lavis_name = lavis_name
        self.model_type = model_type
        self.device_str = device
        self.pooling = pooling
        self.image_size = int(image_size)
    
        # ---- load lavis backbone ----
        from lavis.models import load_model_and_preprocess
        backbone, _, _ = load_model_and_preprocess(
            name=lavis_name, model_type=model_type, is_eval=False, device=device
        )
        self.backbone = backbone
    
        # ---- pick vision module (works for OPT + T5) ----
        if hasattr(self.backbone, "visual_encoder"):
            self.visual = self.backbone.visual_encoder
        elif hasattr(self.backbone, "vision_encoder"):
            self.visual = self.backbone.vision_encoder
        elif hasattr(self.backbone, "visual"):
            self.visual = self.backbone.visual
        else:
            raise AttributeError(f"Cannot find vision encoder on model: {type(self.backbone)}")
    
        # ln_vision exists on blip2_opt (EVA-ViT path)
        self.ln_vision = getattr(self.backbone, "ln_vision", None)
    
        # ---- freeze everything, then selectively unfreeze ----
        _freeze_all(self.backbone)
    
        # ---- detect CLIP-ViT vision (T5-flan vitL uses lavis/models/clip_vit.py) ----
        vision_mod = self.visual.__class__.__module__
        is_clip_vit = (vision_mod.endswith("clip_vit") or ("clip_vit" in vision_mod))
    
        # ---- dtype policy for vision ----
        vit_precision = str(self.cfg.get("vit_precision", "fp16")).lower()
    
        if is_clip_vit:
            # IMPORTANT: CLIP LayerNorm in lavis/models/clip_vit.py casts input to FP32 internally.
            # To avoid LayerNorm / MHA dtype mismatch, keep CLIP vision in FP32.
            self.visual = self.visual.float()
            if self.ln_vision is not None:
                self.ln_vision = self.ln_vision.float()
            self.vision_dtype = torch.float32
        else:
            # EVA-ViT / other vision: can use fp16/bf16/fp32
            if vit_precision in ["fp16", "float16", "16"]:
                self.visual = self.visual.half()
                if self.ln_vision is not None:
                    self.ln_vision = self.ln_vision.float()
                self.vision_dtype = torch.float16
            elif vit_precision in ["bf16", "bfloat16"]:
                self.visual = self.visual.bfloat16()
                if self.ln_vision is not None:
                    self.ln_vision = self.ln_vision.bfloat16()
                self.vision_dtype = torch.bfloat16
            else:
                self.visual = self.visual.float()
                if self.ln_vision is not None:
                    self.ln_vision = self.ln_vision.float()
                self.vision_dtype = torch.float32
    
        # device of vision module
        self.vision_device = next(self.visual.parameters()).device
        print(
            f"[DBG] visual={self.visual.__class__.__name__} dtype/device={self.vision_dtype}/{self.vision_device}",
            flush=True,
        )
    
        # ---- optionally unfreeze vision ----
        if train_vision or int(unfreeze_vision_last_n) > 0:
            if int(unfreeze_vision_last_n) > 0:
                _unfreeze_last_vit_blocks(self.visual, int(unfreeze_vision_last_n))
            else:
                for p in self.visual.parameters():
                    p.requires_grad = True
    
            # keep CLIP vision FP32; for EVA we can keep current dtype or force fp32 if you want stability
            if (not is_clip_vit) and str(self.cfg.get("train_vision_fp32", "0")) in ["1", "true", "yes"]:
                self.visual = self.visual.float()
                self.vision_dtype = torch.float32
    
            if self.ln_vision is not None:
                for p in self.ln_vision.parameters():
                    p.requires_grad = True
    
        # ---- optionally unfreeze qformer / projections ----
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
    
        # ---- head ----
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


    
   
   
    def _pick_proj(self, m):
        """
        Return the best projection layer for mapping Q-Former dim (768)
        to LM hidden size (e.g., OPT 4096, T5 2048), if available.
        """
        for name in ("opt_proj", "t5_proj", "text_proj", "vision_proj", "proj"):
            proj = getattr(m, name, None)
            if proj is not None:
                return proj, name
        return None, None
    
    
    def _encode_image_qformer(self, x: torch.Tensor) -> torch.Tensor:
        m = self.backbone
        visual = self.visual  # set in __init__
        ln_vision = self.ln_vision  # may be None
    
        # ----------------------------
        # 1) Vision forward (always define image_embeds)
        # ----------------------------
        # Ensure input matches vision module dtype/device
        v_p = next(visual.parameters())
        v_dtype, v_device = v_p.dtype, v_p.device
        x = x.to(device=v_device, dtype=v_dtype)
    
        image_embeds = visual(x)  # <-- ALWAYS created here
        # Some vision encoders return dict/tuple; normalize to tensor
        if isinstance(image_embeds, (tuple, list)):
            image_embeds = image_embeds[0]
        if isinstance(image_embeds, dict):
            # pick a common key if it exists
            image_embeds = image_embeds.get("last_hidden_state", None) or image_embeds.get("embeds", None)
    
        if ln_vision is not None:
            # move ln_vision to same device as image_embeds, keep fp32
            ln_vision = ln_vision.to(device=image_embeds.device, dtype=torch.float32)
    
            # feed fp32 to ln_vision (it expects fp32 path)
            image_embeds = ln_vision(image_embeds.float())
    
        #image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device)
    
        qformer = getattr(m, "Qformer", None)
        query_tokens_param = getattr(m, "query_tokens", None)
    
        if qformer is not None and query_tokens_param is not None:
            q_p = next(qformer.parameters())
            q_dtype, q_device = q_p.dtype, q_p.device
    
            image_embeds_q = image_embeds.to(device=q_device, dtype=q_dtype)
    
            # attention mask for encoder (all ones)
            encoder_atts = torch.ones(
                image_embeds_q.size()[:-1], dtype=torch.long, device=q_device
            )
        query_tokens = query_tokens_param.to(device=q_device, dtype=q_dtype)
        query_tokens = query_tokens.expand(image_embeds_q.shape[0], -1, -1)

        query_out = qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds_q,
            encoder_attention_mask=encoder_atts,
            return_dict=True,
        )

        feats = query_out.last_hidden_state  # [B, num_query, hidden]
        return feats
      
        return image_embeds
         


    def _encode_image(self, x: torch.Tensor) -> torch.Tensor:
        # Case A: feature-extractor style models
        if hasattr(self.backbone, "extract_features"):
            out = self.backbone.extract_features({"image": x}, mode="image")

            feats = None
            if hasattr(out, "image_embeds_proj") and out.image_embeds_proj is not None:
                feats = out.image_embeds_proj
            elif hasattr(out, "image_embeds") and out.image_embeds is not None:
                feats = out.image_embeds
            elif isinstance(out, dict):
                feats = out.get("image_embeds_proj") or out.get("image_embeds") or out.get("embeds")

            if feats is None:
                raise RuntimeError(f"extract_features returned unsupported object: {type(out)}")

            if feats.dim() == 3:
                feats = feats.mean(dim=1) if self.pooling == "mean" else feats[:, 0]
            return feats

        # Case B: BLIP2_OPT / BLIP2_T5 unified Q-Former path
        return self._encode_image_qformer(x)
        
    
    def _get_module_dtype_device(m):
        for p in m.parameters():
            return p.dtype, p.device
        # fallback (rare, but safe)
        for b in m.buffers():
            return b.dtype, b.device
        return torch.float32, torch.device(device)
        
    @torch.no_grad()
    def _infer_feature_dim(self):
        visual = self.visual
        vdtype, vdev = _module_dtype_device(visual, fallback_device="cuda")
        dummy = torch.zeros(1, 3, self.image_size, self.image_size,
                        device=self.vision_device, dtype=self.vision_dtype)
        feats = self._encode_image(dummy)
#        dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=self.vision_device, dtype=self.vision_dtype)
#        feats = self._encode_image(dummy)

    
#        dummy = torch.zeros(
#            1, 3, self.image_size, self.image_size,
#            device=vdev, dtype=vdtype
#        )
#        feats = self._encode_image(dummy)
        return feats.shape[-1]

    
#    @torch.no_grad()
#    def _infer_feature_dim(self) -> int:
#        m = self.backbone
#        visual = m.visual_encoder if hasattr(m, "visual_encoder") else m.visual
#    
#        #dummy = torch.randn(1, 3, self.image_size, self.image_size, device="cuda")
#        visual = self.visual_encoder  # or however you reference it
#        vdtype, vdev = _vision_param_dtype_device(visual)
#        #dummy = torch.randn(1, 3, self.image_size, self.image_size, device=vdev, dtype=vdtype)
#        #dummy = torch.zeros(1, 3, self.image_size, self.image_size,device=self.vision_device, dtype=self.vision_dtype)
#        dummy = torch.zeros(1, 3, self.image_size, self.image_size, device=self.vision_device, dtype=self.vision_dtype)
#
#        dummy = _cast_image_like_vision(dummy, visual)  # ✅ make dummy match conv1/patch dtype
#    
#        feats = self._encode_image(dummy)
#        return feats.shape[-1]


    
#    def forward(self, x: torch.Tensor):
#        # x is the image tensor [B,3,H,W]
#        feats = self._encode_image(x)   # unified encoder
#        return self.classifier(feats)

    def forward(self, images):
        feats = self._encode_image(images)  # could be [B, T, D] or [B, D]
    
        # If Qformer returns token features, pool them to one vector per image
        if feats.dim() == 3:  # [B, T, D]
            if self.pooling == "cls":
                feats = feats[:, 0]            # [B, D]
            else:
                feats = feats.mean(dim=1)      # [B, D]
    
        logits = self.classifier(feats)        # [B, num_classes]
        return logits

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
        # Project to LLM hidden size if available (OPT / T5 / Vicuna, etc.)
        proj = (
            getattr(self.backbone, "opt_proj", None)
            or getattr(self.backbone, "t5_proj", None)
            or getattr(self.backbone, "llm_proj", None)
            or getattr(self.backbone, "proj", None)
        )
        if proj is not None:
            pooled = proj(pooled)
        return pooled


# Backward-compatible name used by older code/configs.
class Blip2MultiLabelClassifier(Blip2Classifier):
    pass
