"""
Analytical and High-Precision Finite Difference Ground Truth for Conjugate Heat Transfer.
Solves coupled conjugate 2D thermal conduction and convection with sharp interface jump.
"""

import numpy as np


def solve_conjugate_ground_truth(
    nx: int = 80,
    ny_solid: int = 40,
    ny_fluid: int = 40,
    L: float = 2.0,
    H_s: float = 0.5,
    H_f: float = 0.5,
    k_ratio: float = 10.0,
    pe: float = 10.0,
    max_iter: int = 3000,
    tol: float = 1e-6,
):
    """
    Solves 2D CHT on structured grid using Conservative Finite Volume Method.
    Solid domain: y in [-H_s, 0]
    Fluid domain: y in [0, H_f]
    """
    x = np.linspace(0, L, nx)
    y_s = np.linspace(-H_s, 0, ny_solid)
    y_f = np.linspace(0, H_f, ny_fluid)
    dx = L / (nx - 1)
    dys = H_s / (ny_solid - 1)
    dyf = H_f / (ny_fluid - 1)

    T_s = np.zeros((nx, ny_solid))
    T_f = np.zeros((nx, ny_fluid))

    # Boundary conditions:
    # Solid bottom (y = -H_s): uniform heat flux or fixed high temperature (e.g. T = 1.0)
    T_s[:, 0] = 1.0
    # Fluid inlet (x = 0): T_f = 0.0 (cold coolant)
    T_f[0, :] = 0.0
    # Fluid top (y = H_f): adiabatic (dT/dy = 0)

    # Fully developed parabolic velocity in fluid
    u_vel = 6.0 * (y_f / H_f) * (1.0 - y_f / H_f)

    # Gauss-Seidel iterative relaxation with harmonic interface flux coupling
    for it in range(max_iter):
        T_s_old = T_s.copy()
        T_f_old = T_f.copy()

        # 1. Solid Interior Conduction: d2T/dx2 + d2T/dy2 = 0
        for i in range(1, nx - 1):
            for j in range(1, ny_solid - 1):
                T_s[i, j] = 0.5 * (
                    (T_s[i + 1, j] + T_s[i - 1, j]) / dx**2 +
                    (T_s[i, j + 1] + T_s[i, j - 1]) / dys**2
                ) / (1.0 / dx**2 + 1.0 / dys**2)

        # 2. Fluid Interior Convection: u * dT/dx = (1/Pe) * (d2T/dx2 + d2T/dy2)
        for i in range(1, nx - 1):
            for j in range(1, ny_fluid - 1):
                u = u_vel[j]
                # Upwind for convection + central for diffusion
                conv = u * T_f[i - 1, j] / dx
                diff = (1.0 / pe) * (
                    (T_f[i + 1, j] + T_f[i - 1, j]) / dx**2 +
                    (T_f[i, j + 1] + T_f[i, j - 1]) / dyf**2
                )
                denom = u / dx + (2.0 / pe) * (1.0 / dx**2 + 1.0 / dyf**2)
                T_f[i, j] = (conv + diff) / denom

        # 3. Interface Coupling at y = 0: ks * (Ts[j] - Ts[j-1])/dys = kf * (Tf[1] - Tf[0])/dyf
        # and Ts[:, -1] == Tf[:, 0]
        T_int = (k_ratio * T_s[:, -2] / dys + T_f[:, 1] / dyf) / (k_ratio / dys + 1.0 / dyf)
        T_s[:, -1] = T_int
        T_f[:, 0] = T_int

        # Boundary conditions at ends
        T_s[0, :] = T_s[1, :]   # Insulated solid inlet
        T_s[-1, :] = T_s[-2, :] # Insulated solid outlet
        T_f[-1, :] = T_f[-2, :] # Outflow Neumann
        T_f[:, -1] = T_f[:, -2] # Top adiabatic

        err = np.max(np.abs(T_s - T_s_old)) + np.max(np.abs(T_f - T_f_old))
        if err < tol:
            break

    return {
        "x": x,
        "y_solid": y_s,
        "y_fluid": y_f,
        "T_solid": T_s,
        "T_fluid": T_f,
        "T_interface": T_s[:, -1],
    }
