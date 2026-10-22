"""
Monolithic (Single-Network) PINN Baseline.
Illustrates spectral bias and failure to capture discontinuous normal heat flux across sharp material interfaces.
"""

import torch
import torch.nn as nn
from models.xpinn_module import SubnetMLP


class MonolithicPINN(nn.Module):
    """Single monolithic network attempting to model both solid and fluid domains concurrently."""

    def __init__(self, hidden_dim: int = 64, num_layers: int = 5):
        super(MonolithicPINN, self).__init__()
        self.net = SubnetMLP(in_features=2, hidden_dim=hidden_dim, num_layers=num_layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.net(x, y)
