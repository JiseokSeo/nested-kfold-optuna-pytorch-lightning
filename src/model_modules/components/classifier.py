# src/models/classifier.py
import torch
import torch.nn as nn
from typing import Optional, List
from src.utils import generate_layer_dims


class MLPClassifier(nn.Module):
    """여러 레이어를 지원하는 MLP 분류기 (config dict 기반 생성)"""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.name = config.get("name", "mlp")
        self.params = config.get("params", {})
        self.input_dim = self.params.get("input_dim")
        self.num_classes = self.params.get("num_classes")
        self.n_layers = self.params.get("n_layers")
        self.n_units = self.params.get("n_units")
        self.output_dim = self.params.get("output_dim")
        self.structure = self.params.get("structure")
        self.dropout = self.params.get(
            "dropout",
        )
        self.activation = self.params.get("activation")
        self.use_norm = self.params.get("use_norm", True)

        # activation 함수 매핑
        act_map = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}
        act_cls = act_map.get(self.activation.lower(), nn.ReLU)

        # hidden_dims 생성
        hidden_dims = generate_layer_dims(
            self.input_dim,
            self.n_layers,
            self.n_units,
            self.output_dim,
            self.structure,
        )
        # 마지막은 num_classes로 강제
        hidden_dims = hidden_dims[:-1] + [self.num_classes]

        layers = []
        prev_dim = self.input_dim
        for i, h_dim in enumerate(hidden_dims):
            layers.append(nn.Linear(prev_dim, h_dim))
            if i < len(hidden_dims) - 1:  # 마지막 레이어 전까지만
                if self.use_norm:
                    layers.append(nn.BatchNorm1d(h_dim))
                layers.append(act_cls())
                layers.append(nn.Dropout(self.dropout))
            prev_dim = h_dim
        self.mlp = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.mlp(x)
