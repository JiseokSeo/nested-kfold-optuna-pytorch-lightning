# src/models/csv_extractor.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Type
from src.utils import generate_layer_dims


# --- Basic MLP Block ---
class BasicMLPBlock(nn.Module):
    """A basic MLP block: Linear -> (BatchNorm) -> Activation -> Dropout"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.2,
        use_norm: bool = True,
        activation: str = "silu",
    ):
        super().__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = (
            nn.BatchNorm1d(out_dim) if use_norm and out_dim > 0 else nn.Identity()
        )
        self.act = nn.SiLU() if activation == "silu" else nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        if isinstance(self.norm, nn.BatchNorm1d):
            if x.ndim == 1:
                x = self.norm(x.unsqueeze(0)).squeeze(0)
            elif x.shape[0] > 1:
                x = self.norm(x)
            # else: batch size 1, ndim > 1: skip norm
        x = self.act(x)
        x = self.dropout(x)
        return x


# --- Residual MLP Block ---
class ResidualMLPBlock(nn.Module):
    """An MLP block with a residual connection: Input -> [Main Path + Shortcut Path]"""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        dropout: float = 0.2,
        use_norm: bool = True,
        activation: str = "silu",
    ):
        super().__init__()
        # Main path
        self.fc = nn.Linear(in_dim, out_dim)
        self.norm = (
            nn.BatchNorm1d(out_dim) if use_norm and out_dim > 0 else nn.Identity()
        )
        self.act = nn.SiLU() if activation == "silu" else nn.ReLU()
        self.dropout = nn.Dropout(dropout)

        # Shortcut connection
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.shortcut(x)

        # Main path computation
        out = self.fc(x)
        if isinstance(self.norm, nn.BatchNorm1d):
            if out.ndim == 1:
                out = self.norm(out.unsqueeze(0)).squeeze(0)
            elif out.shape[0] > 1:
                out = self.norm(out)
            # else: batch size 1, ndim > 1: skip norm
        out = self.act(out)
        out = self.dropout(out)

        # Add residual
        out += residual
        # Optional: Activation after addition
        # out = self.act(out) # 필요하다면 residual 합 이후에 활성화 함수를 추가할 수 있습니다.
        return out


# --- Unified MLP Extractor ---
class MLPCSVExtractor(nn.Module):
    """
    MLP Feature Extractor for CSV data, capable of using Basic or Residual blocks.
    The choice of block is determined by the `use_residual` flag.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = config.get("name")
        self.params = config.get("params")
        self.input_dim = self.params.get("input_dim")
        self.n_layers = self.params.get("n_layers")
        self.n_units = self.params.get("n_units")
        self.structure = self.params.get("structure")
        self.dropout = self.params.get("dropout")
        self.output_dim = self.params.get("output_dim")
        self.activation = self.params.get("activation")
        self.use_norm = self.params.get("use_norm")

        BlockType = ResidualMLPBlock if self.name == "res" else BasicMLPBlock

        layers = []
        prev_dim = self.input_dim
        hidden_dims = generate_layer_dims(
            self.input_dim, self.n_layers, self.n_units, self.output_dim, self.structure
        )
        for i, h_dim in enumerate(hidden_dims):
            layers.append(
                BlockType(prev_dim, h_dim, self.dropout, self.use_norm, self.activation)
            )
            prev_dim = h_dim  # Output of current layer is input to next

        self.mlp = nn.Sequential(*layers)
        self.output_dim = prev_dim  # The final dimension after the last layer

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
