"""Loss functions for nulling-angle prediction."""

from __future__ import annotations

import torch
from torch import nn


class NullingAngleLoss(nn.Module):
    """MAE prediction loss plus a soft mainlobe-protection constraint.

    The model predicts normalized nulling angles in [0, 1], corresponding to
    [-90, 90] degrees. The first two input features are normalized steering
    angles in [0, 1], corresponding to [-40, 40] degrees.
    """

    def __init__(self, min_separation_deg: float = 15.0, constraint_weight: float = 0.05):
        super().__init__()
        self.min_separation_deg = min_separation_deg
        self.constraint_weight = constraint_weight
        self.mae = nn.L1Loss()

    @staticmethod
    def _denorm_steering(x: torch.Tensor) -> torch.Tensor:
        return x * 80.0 - 40.0

    @staticmethod
    def _denorm_nulling(y: torch.Tensor) -> torch.Tensor:
        return y * 180.0 - 90.0

    def forward(self, pred: torch.Tensor, target: torch.Tensor, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        main_loss = self.mae(pred, target)

        theta_d = self._denorm_steering(inputs[:, 0])
        theta_u = self._denorm_steering(inputs[:, 1])
        pred_deg = self._denorm_nulling(pred)

        sep_d = torch.abs(pred_deg[:, 0] - theta_d)
        sep_u = torch.abs(pred_deg[:, 1] - theta_u)
        constraint = torch.relu(self.min_separation_deg - sep_d) + torch.relu(self.min_separation_deg - sep_u)
        constraint_loss = constraint.mean() / 180.0

        total_loss = main_loss + self.constraint_weight * constraint_loss
        return total_loss, main_loss.detach(), constraint_loss.detach()
