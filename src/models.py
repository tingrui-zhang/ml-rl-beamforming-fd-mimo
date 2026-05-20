"""Neural network models for nulling-angle prediction."""

from __future__ import annotations

import torch
from torch import nn


class DNNNullingPredictor(nn.Module):
    """Fully connected predictor inspired by the thesis/paper DNN pipeline."""

    def __init__(self, input_dim: int = 6, output_dim: int = 2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class CNNNullingPredictor(nn.Module):
    """Lightweight 1D CNN predictor for six engineered angular features."""

    def __init__(self, input_channels: int = 6, output_dim: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Accept either [batch, 6] or [batch, 6, 1].
        if x.ndim == 2:
            x = x.unsqueeze(-1)
        return self.head(self.features(x))


def build_model(model_name: str) -> nn.Module:
    model_name = model_name.lower()
    if model_name == "dnn":
        return DNNNullingPredictor()
    if model_name == "cnn":
        return CNNNullingPredictor()
    raise ValueError(f"Unsupported model_name={model_name!r}. Choose 'dnn' or 'cnn'.")
