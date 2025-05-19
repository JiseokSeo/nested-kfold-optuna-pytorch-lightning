# src/models/fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ConcatFusion(nn.Module):
    """Simple concatenation fusion."""

    def __init__(self, config):
        super().__init__()
        params = config.get("params", {})
        self.image_dim = params["image_dim"]
        self.csv_dim = params["csv_dim"]
        self.output_dim = self.image_dim + self.csv_dim

    def forward(
        self, image_features: torch.Tensor, csv_features: torch.Tensor
    ) -> torch.Tensor:
        return torch.cat([image_features, csv_features], dim=1)


class GatingFusion(nn.Module):
    """Gating mechanism for fusion."""

    def __init__(self, config):
        super().__init__()
        params = config.get("params", {})
        self.image_dim = params["image_dim"]
        self.csv_dim = params["csv_dim"]
        self.gate_dim = params.get("gate_dim", self.csv_dim)
        self.gate = nn.Sequential(
            nn.Linear(self.image_dim, self.gate_dim), nn.Sigmoid()
        )
        self.output_dim = self.image_dim + self.csv_dim

    def forward(
        self, image_features: torch.Tensor, csv_features: torch.Tensor
    ) -> torch.Tensor:
        gate_values = self.gate(image_features)
        if gate_values.shape[1] == csv_features.shape[1]:
            gated_csv = csv_features * gate_values
        else:
            print(
                f"Warning: Gate dim ({gate_values.shape[1]}) != CSV dim ({csv_features.shape[1]}). Gating might be incorrect."
            )
            gated_csv = csv_features
        return torch.cat([image_features, gated_csv], dim=1)


# Add FiLM, Attention based fusion etc. as needed


class FusionLayer(nn.Module):
    """Config 기반으로 적절한 fusion 모듈을 내부적으로 선택하는 래퍼."""

    def __init__(self, config):
        super().__init__()
        name = config.get("name", "concat").lower()
        if name == "concat":
            self.fusion = ConcatFusion(config)
        elif name == "gating":
            self.fusion = GatingFusion(config)
        else:
            raise ValueError(f"지원하지 않는 fusion type: {name}")
        self.output_dim = self.fusion.output_dim

    def forward(self, image_features, csv_features):
        return self.fusion(image_features, csv_features)
