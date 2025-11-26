"""
GeoGuessr classifier built on top of OpenCLIP vision backbones.
"""
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import open_clip


@dataclass
class ModelConfig:
    model_name: str = "ViT-H-14"
    pretrained: str = "laion2b_s32b_b79k"
    num_classes: int = 1024
    dropout: float = 0.1
    freeze_backbone: bool = False
    normalize_features: bool = True
    compile_model: bool = False


class GeoGuessrModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        clip_model, _, preprocess = open_clip.create_model_and_transforms(cfg.model_name, pretrained=cfg.pretrained)
        self.preprocess = preprocess
        self.visual = clip_model.visual
        embed_dim = clip_model.visual.output_dim
        self.head = nn.Sequential(nn.LayerNorm(embed_dim), nn.Dropout(cfg.dropout), nn.Linear(embed_dim, cfg.num_classes))
        self.normalize_features = cfg.normalize_features

        if cfg.freeze_backbone:
            for p in self.visual.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        feats = self.visual(images)
        feats = feats.float()
        if self.normalize_features:
            feats = F.normalize(feats, dim=-1)
        logits = self.head(feats)
        return logits, feats


def build_model(cfg: ModelConfig, device: torch.device) -> GeoGuessrModel:
    model = GeoGuessrModel(cfg)
    model = model.to(device)
    if cfg.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
    return model
