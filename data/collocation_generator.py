"""
Collocation Point Generator for Domain Decomposition XPINN.
Samples interior solid, interior fluid, boundary, and interface collocation sets.
"""

import torch
import numpy as np


def generate_cht_collocation(
    n_solid: int = 1500,
    n_fluid: int = 1500,
    n_interface: int = 400,
    n_boundary: int = 300,
    L: float = 2.0,
    H_s: float = 0.5,
    H_f: float = 0.5,
    seed: int = 42,
):
    """Generates Latin Hypercube or uniform random collocation point tensors."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 1. Solid Interior (y in [-H_s, 0])
    x_s = torch.rand(n_solid, 1) * L
    y_s = torch.rand(n_solid, 1) * (-H_s)

    # 2. Fluid Interior (y in [0, H_f])
    x_f = torch.rand(n_fluid, 1) * L
    y_f = torch.rand(n_fluid, 1) * H_f

    # 3. Interface (y = 0)
    x_int = torch.rand(n_interface, 1) * L
    y_int = torch.zeros(n_interface, 1)

    # 4. Solid Heated Bottom (y = -H_s, T = 1.0)
    x_bot = torch.rand(n_boundary, 1) * L
    y_bot = torch.full((n_boundary, 1), -H_s)
    T_bot = torch.ones(n_boundary, 1)

    # 5. Fluid Cold Inlet (x = 0, y in [0, H_f], T = 0.0)
    x_inlet = torch.zeros(n_boundary, 1)
    y_inlet = torch.rand(n_boundary, 1) * H_f
    T_inlet = torch.zeros(n_boundary, 1)

    # 6. Fluid Adiabatic Top (y = H_f, dT/dy = 0)
    x_top = torch.rand(n_boundary, 1) * L
    y_top = torch.full((n_boundary, 1), H_f)

    return {
        "solid": (x_s, y_s),
        "fluid": (x_f, y_f),
        "interface": (x_int, y_int),
        "bottom": (x_bot, y_bot, T_bot),
        "inlet": (x_inlet, y_inlet, T_inlet),
        "top": (x_top, y_top),
    }
