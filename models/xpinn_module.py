"""
Extended Physics-Informed Neural Network (XPINN) Architecture.
Based on Jagtap & Karniadakis (2020), "Extended Physics-Informed Neural Networks (XPINNs)".
Implements decoupled neural sub-networks for solid and fluid subdomains communicating via transmission losses.
"""

import torch
import torch.nn as nn


class SubnetMLP(nn.Module):
    """Deep fully connected sub-network for a specific physics subdomain."""

    def __init__(self, in_features: int = 2, hidden_dim: int = 64, num_layers: int = 4):
        super(SubnetMLP, self).__init__()
        layers = []
        layers.append(nn.Linear(in_features, hidden_dim))
        layers.append(nn.Tanh())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([x, y], dim=-1)
        return self.net(inputs)


class XPINNConjugate(nn.Module):
    """
    Two-domain XPINN for Conjugate Heat Transfer:
      - Solid sub-network: N_s(x, y) on Omega_s
      - Fluid sub-network: N_f(x, y) on Omega_f
      - Communication exclusively enforced via interface transmission loss
    """

    def __init__(self, hidden_dim: int = 64, num_layers: int = 4):
        super(XPINNConjugate, self).__init__()
        self.solid_net = SubnetMLP(in_features=2, hidden_dim=hidden_dim, num_layers=num_layers)
        self.fluid_net = SubnetMLP(in_features=2, hidden_dim=hidden_dim, num_layers=num_layers)

    def forward_solid(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.solid_net(x, y)

    def forward_fluid(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.fluid_net(x, y)
