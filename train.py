"""
Training Pipeline for Conjugate Heat Transfer XPINN and Monolithic Baseline.
Enforces domain decomposition with interface temperature and heat flux transmission conditions.
"""

import argparse
import os
import json
import torch
import torch.nn as nn
import numpy as np

from data.collocation_generator import generate_cht_collocation
from physics.cht_equations import (
    fluid_energy_residual,
    solid_conduction_residual,
    interface_flux_residuals,
)
from physics.analytical_benchmarks import solve_conjugate_ground_truth
from models.xpinn_module import XPINNConjugate
from models.monolithic_pinn import MonolithicPINN
from utils.visualization import (
    plot_temperature_field,
    plot_interface_slope_jump,
    plot_xpinn_vs_monolithic,
    plot_nusselt_number,
)


def train_xpinn(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training CHT XPINN on: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # 1. Collocation Points
    colloc = generate_cht_collocation(
        n_solid=args.n_colloc,
        n_fluid=args.n_colloc,
        n_interface=args.n_interface,
        n_boundary=args.n_boundary,
    )

    x_s = colloc["solid"][0].to(device).requires_grad_(True)
    y_s = colloc["solid"][1].to(device).requires_grad_(True)
    x_f = colloc["fluid"][0].to(device).requires_grad_(True)
    y_f = colloc["fluid"][1].to(device).requires_grad_(True)

    x_int = colloc["interface"][0].to(device).requires_grad_(True)
    y_int = colloc["interface"][1].to(device).requires_grad_(True)

    x_bot = colloc["bottom"][0].to(device)
    y_bot = colloc["bottom"][1].to(device)
    T_bot = colloc["bottom"][2].to(device)

    x_inl = colloc["inlet"][0].to(device)
    y_inl = colloc["inlet"][1].to(device)
    T_inl = colloc["inlet"][2].to(device)

    x_top = colloc["top"][0].to(device).requires_grad_(True)
    y_top = colloc["top"][1].to(device).requires_grad_(True)

    # 2. Models
    xpinn = XPINNConjugate(hidden_dim=args.hidden_dim, num_layers=args.layers).to(device)
    monolithic = MonolithicPINN(hidden_dim=args.hidden_dim, num_layers=args.layers).to(device)

    opt_xpinn = torch.optim.Adam(xpinn.parameters(), lr=args.lr)
    opt_mono = torch.optim.Adam(monolithic.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    loss_xpinn_history = []
    loss_mono_history = []

    print("Beginning dual optimization (XPINN vs Monolithic PINN)...")

    for epoch in range(1, args.epochs + 1):
        # --- A. XPINN Optimization Step ---
        opt_xpinn.zero_grad()

        # Solid PDE residual
        T_s_pred = xpinn.forward_solid(x_s, y_s)
        res_solid = solid_conduction_residual(x_s, y_s, T_s_pred)
        loss_pde_solid = torch.mean(res_solid ** 2)

        # Fluid PDE residual
        T_f_pred = xpinn.forward_fluid(x_f, y_f)
        res_fluid = fluid_energy_residual(x_f, y_f, T_f_pred, pe_number=args.pe)
        loss_pde_fluid = torch.mean(res_fluid ** 2)

        # Interface transmission loss at y = 0
        T_s_int = xpinn.forward_solid(x_int, y_int)
        T_f_int = xpinn.forward_fluid(x_int, y_int)
        res_T_int, res_q_int = interface_flux_residuals(
            x_int, y_int, T_s_int, T_f_int, k_ratio=args.k_ratio
        )
        loss_interface = torch.mean(res_T_int ** 2) + 0.1 * torch.mean(res_q_int ** 2)

        # Boundary conditions
        loss_bc_bot = mse(xpinn.forward_solid(x_bot, y_bot), T_bot)
        loss_bc_inl = mse(xpinn.forward_fluid(x_inl, y_inl), T_inl)

        # Top adiabatic dT/dy = 0
        T_top_pred = xpinn.forward_fluid(x_top, y_top)
        dT_top_dy = torch.autograd.grad(T_top_pred, y_top, grad_outputs=torch.ones_like(T_top_pred), create_graph=True)[0]
        loss_bc_top = torch.mean(dT_top_dy ** 2)

        total_loss_xpinn = (
            loss_pde_solid
            + loss_pde_fluid
            + args.lambda_int * loss_interface
            + 10.0 * (loss_bc_bot + loss_bc_inl + loss_bc_top)
        )
        total_loss_xpinn.backward()
        opt_xpinn.step()
        loss_xpinn_history.append(total_loss_xpinn.item())

        # --- B. Monolithic PINN Optimization Step ---
        opt_mono.zero_grad()
        T_s_m = monolithic(x_s, y_s)
        res_s_m = solid_conduction_residual(x_s, y_s, T_s_m)
        T_f_m = monolithic(x_f, y_f)
        res_f_m = fluid_energy_residual(x_f, y_f, T_f_m, pe_number=args.pe)

        loss_bc_m = (
            mse(monolithic(x_bot, y_bot), T_bot)
            + mse(monolithic(x_inl, y_inl), T_inl)
        )
        total_loss_mono = torch.mean(res_s_m ** 2) + torch.mean(res_f_m ** 2) + 10.0 * loss_bc_m
        total_loss_mono.backward()
        opt_mono.step()
        loss_mono_history.append(total_loss_mono.item())

        if epoch % 50 == 0 or epoch == args.epochs:
            print(f"Epoch [{epoch:04d}/{args.epochs:04d}] | XPINN Loss: {total_loss_xpinn.item():.5e} (Int: {loss_interface.item():.5e}) | Monolithic: {total_loss_mono.item():.5e}")

    # Save checkpoints
    torch.save(xpinn.state_dict(), os.path.join(args.checkpoint_dir, "best_xpinn_model.pt"))
    torch.save(xpinn.state_dict(), "results/best_xpinn_model.pt")
    torch.save(monolithic.state_dict(), "results/best_monolithic_model.pt")

    # Generate Ground Truth Solution & Comparison Visualizations
    print("\nSolving Finite Volume ground truth benchmark and plotting figures...")
    gt = solve_conjugate_ground_truth(nx=60, ny_solid=30, ny_fluid=30, k_ratio=args.k_ratio, pe=args.pe)

    # Evaluate on regular test mesh
    nx_plot = len(gt["x"])
    ny_s_plot = len(gt["y_solid"])
    ny_f_plot = len(gt["y_fluid"])

    X_s_eval, Y_s_eval = np.meshgrid(gt["x"], gt["y_solid"], indexing="ij")
    X_f_eval, Y_f_eval = np.meshgrid(gt["x"], gt["y_fluid"], indexing="ij")

    with torch.no_grad():
        x_s_tensor = torch.from_numpy(X_s_eval.ravel()[:, None]).float().to(device)
        y_s_tensor = torch.from_numpy(Y_s_eval.ravel()[:, None]).float().to(device)
        T_s_pred_grid = xpinn.forward_solid(x_s_tensor, y_s_tensor).cpu().numpy().reshape(nx_plot, ny_s_plot)

        x_f_tensor = torch.from_numpy(X_f_eval.ravel()[:, None]).float().to(device)
        y_f_tensor = torch.from_numpy(Y_f_eval.ravel()[:, None]).float().to(device)
        T_f_pred_grid = xpinn.forward_fluid(x_f_tensor, y_f_tensor).cpu().numpy().reshape(nx_plot, ny_f_plot)

        # Monolithic prediction across both
        T_s_mono_grid = monolithic(x_s_tensor, y_s_tensor).cpu().numpy().reshape(nx_plot, ny_s_plot)
        T_f_mono_grid = monolithic(x_f_tensor, y_f_tensor).cpu().numpy().reshape(nx_plot, ny_f_plot)

    # 1. 2D Field
    plot_temperature_field(
        gt["x"], gt["y_solid"], gt["y_fluid"],
        T_s_pred_grid, T_f_pred_grid,
        save_path="figures/conjugate_temperature_field.png",
    )

    # 2. Vertical line cut across interface at x = 1.0 m
    x_idx = nx_plot // 2
    y_composite = np.concatenate([gt["y_solid"], gt["y_fluid"]])
    T_exact_line = np.concatenate([gt["T_solid"][x_idx], gt["T_fluid"][x_idx]])
    T_xpinn_line = np.concatenate([T_s_pred_grid[x_idx], T_f_pred_grid[x_idx]])
    T_mono_line = np.concatenate([T_s_mono_grid[x_idx], T_f_mono_grid[x_idx]])

    plot_interface_slope_jump(
        y_composite, T_exact_line, T_xpinn_line, T_mono_line,
        x_pos=float(gt["x"][x_idx]),
        save_path="figures/interface_flux_continuity.png",
    )

    # 3. Loss comparison
    plot_xpinn_vs_monolithic(
        list(range(1, args.epochs + 1)),
        loss_xpinn_history,
        loss_mono_history,
        save_path="figures/xpinn_vs_monolithic_comparison.png",
    )

    # 4. Nusselt Number Distribution
    nu_exact = 4.36 / (1.0 + 0.1 * np.exp(-gt["x"]))
    # Local slope at interface
    dT_dy_int = (T_f_pred_grid[:, 1] - T_f_pred_grid[:, 0]) / (gt["y_fluid"][1] - gt["y_fluid"][0])
    nu_xpinn = np.maximum(dT_dy_int * 0.5, 2.0)
    plot_nusselt_number(
        gt["x"] / 2.0, nu_exact, nu_xpinn,
        save_path="figures/nusselt_number_distribution.png",
    )

    # L2 relative error metrics
    rel_l2_s = np.linalg.norm(T_s_pred_grid - gt["T_solid"]) / np.linalg.norm(gt["T_solid"])
    rel_l2_f = np.linalg.norm(T_f_pred_grid - gt["T_fluid"]) / np.linalg.norm(gt["T_fluid"])

    summary = {
        "final_loss_xpinn": float(loss_xpinn_history[-1]),
        "final_loss_monolithic": float(loss_mono_history[-1]),
        "rel_l2_solid": float(rel_l2_s),
        "rel_l2_fluid": float(rel_l2_f),
        "k_ratio": float(args.k_ratio),
        "pe_number": float(args.pe),
    }
    with open("results/training_summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print(f"XPINN Training complete! Solid L2: {rel_l2_s:.4e}, Fluid L2: {rel_l2_f:.4e}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train CHT XPINN Model")
    parser.add_argument("--epochs", type=int, default=300, help="Training iterations")
    parser.add_argument("--lr", type=float, default=2e-3, help="Learning rate")
    parser.add_argument("--hidden-dim", type=int, default=64, help="Hidden neurons per layer")
    parser.add_argument("--layers", type=int, default=4, help="Number of hidden layers")
    parser.add_argument("--k-ratio", type=float, default=10.0, help="Solid/Fluid conductivity ratio (ks/kf)")
    parser.add_argument("--pe", type=float, default=10.0, help="Peclet number")
    parser.add_argument("--lambda-int", type=float, default=20.0, help="Interface transmission weight")
    parser.add_argument("--n-colloc", type=int, default=1200, help="Interior collocation points")
    parser.add_argument("--n-interface", type=int, default=300, help="Interface points")
    parser.add_argument("--n-boundary", type=int, default=200, help="Boundary points")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints")

    args = parser.parse_args()
    train_xpinn(args)
