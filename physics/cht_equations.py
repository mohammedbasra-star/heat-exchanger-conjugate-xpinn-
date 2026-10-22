"""
Governing Equations for 2D Conjugate Heat Transfer (CHT).
Couples convective heat transport in fluid channels with thermal conduction in solid metallic walls.
"""

import torch


def fluid_energy_residual(
    x: torch.Tensor,
    y: torch.Tensor,
    T_f: torch.Tensor,
    pe_number: float = 10.0,
    H_f: float = 1.0,
):
    """
    Computes fluid convective-diffusion energy PDE residual:
    u(y) * dT/dx - (1 / Pe) * (d2T/dx2 + d2T/dy2) = 0
    Parabolic laminar velocity profile: u(y) = 6 * (y / H_f) * (1 - y / H_f)
    """
    # 1st derivatives via autograd
    dT_dx = torch.autograd.grad(T_f, x, grad_outputs=torch.ones_like(T_f), create_graph=True)[0]
    dT_dy = torch.autograd.grad(T_f, y, grad_outputs=torch.ones_like(T_f), create_graph=True)[0]

    # 2nd derivatives
    d2T_dx2 = torch.autograd.grad(dT_dx, x, grad_outputs=torch.ones_like(dT_dx), create_graph=True)[0]
    d2T_dy2 = torch.autograd.grad(dT_dy, y, grad_outputs=torch.ones_like(dT_dy), create_graph=True)[0]

    # Poiseuille velocity profile
    y_norm = torch.clamp(y / H_f, min=0.0, max=1.0)
    u_vel = 6.0 * y_norm * (1.0 - y_norm)

    laplacian_T = d2T_dx2 + d2T_dy2
    residual = u_vel * dT_dx - (1.0 / pe_number) * laplacian_T
    return residual


def solid_conduction_residual(
    x: torch.Tensor,
    y: torch.Tensor,
    T_s: torch.Tensor,
):
    """
    Computes solid steady-state Laplace thermal conduction residual:
    d2Ts/dx2 + d2Ts/dy2 = 0
    """
    dTs_dx = torch.autograd.grad(T_s, x, grad_outputs=torch.ones_like(T_s), create_graph=True)[0]
    dTs_dy = torch.autograd.grad(T_s, y, grad_outputs=torch.ones_like(T_s), create_graph=True)[0]

    d2Ts_dx2 = torch.autograd.grad(dTs_dx, x, grad_outputs=torch.ones_like(dTs_dx), create_graph=True)[0]
    d2Ts_dy2 = torch.autograd.grad(dTs_dy, y, grad_outputs=torch.ones_like(dTs_dy), create_graph=True)[0]

    residual = d2Ts_dx2 + d2Ts_dy2
    return residual


def interface_flux_residuals(
    x_int: torch.Tensor,
    y_int: torch.Tensor,
    T_s_int: torch.Tensor,
    T_f_int: torch.Tensor,
    k_ratio: float = 10.0,
):
    """
    Evaluates interface transmission conditions at y = 0:
    1. Temperature continuity: Ts(x, 0) - Tf(x, 0) = 0
    2. Heat flux continuity: -ks * (dTs/dy) = -kf * (dTf/dy) => (dTf/dy) - k_ratio * (dTs/dy) = 0
    """
    dTs_dy = torch.autograd.grad(T_s_int, y_int, grad_outputs=torch.ones_like(T_s_int), create_graph=True)[0]
    dTf_dy = torch.autograd.grad(T_f_int, y_int, grad_outputs=torch.ones_like(T_f_int), create_graph=True)[0]

    res_temperature = T_s_int - T_f_int
    res_flux = dTf_dy - k_ratio * dTs_dy

    return res_temperature, res_flux
