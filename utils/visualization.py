"""
Publication-Grade Visualizations for Conjugate Heat Transfer and XPINNs.
Produces 300 DPI figures formatted for International Journal of Heat and Mass Transfer.
"""

import os
import matplotlib.pyplot as plt
import numpy as np


def set_publication_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10,
        "figure.titlesize": 14,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
    })


def plot_temperature_field(
    x_grid: np.ndarray,
    y_solid: np.ndarray,
    y_fluid: np.ndarray,
    T_solid: np.ndarray,
    T_fluid: np.ndarray,
    save_path: str = "figures/conjugate_temperature_field.png",
):
    set_publication_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))

    # Composite contour
    X_s, Y_s = np.meshgrid(x_grid, y_solid, indexing="ij")
    X_f, Y_f = np.meshgrid(x_grid, y_fluid, indexing="ij")

    c1 = ax.contourf(X_s, Y_s, T_solid, levels=40, cmap="hot")
    c2 = ax.contourf(X_f, Y_f, T_fluid, levels=40, cmap="hot")

    # Mark the physical interface at y = 0
    ax.axhline(0.0, color="cyan", linestyle="--", lw=2, label="Solid-Fluid Interface (y = 0)")
    ax.text(0.1, -0.25, "Solid Metallic Wall ($\Omega_s$)", color="white", fontweight="bold", fontsize=12)
    ax.text(0.1, 0.25, "Fluid Cooling Channel ($\Omega_f$)", color="cyan", fontweight="bold", fontsize=12)

    ax.set_xlabel("Axial Length x (m)")
    ax.set_ylabel("Transverse Height y (m)")
    ax.set_title("XPINN 2D Conjugate Temperature Field with Sharp Interface Gradient")
    cbar = plt.colorbar(c2, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Normalized Temperature T")
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Temperature field saved to {save_path}")


def plot_interface_slope_jump(
    y_line: np.ndarray,
    T_exact: np.ndarray,
    T_xpinn: np.ndarray,
    T_monolithic: np.ndarray,
    x_pos: float = 1.0,
    save_path: str = "figures/interface_flux_continuity.png",
):
    set_publication_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(y_line, T_exact, "k-", lw=2.5, label="Exact FVM Benchmark")
    ax.plot(y_line, T_xpinn, "r--", lw=2, label="XPINN (Domain Decomposition)")
    ax.plot(y_line, T_monolithic, "b:", lw=1.8, label="Monolithic PINN (Smeared)")

    ax.axvline(0.0, color="gray", linestyle="-.", lw=1.2, label="Interface y = 0")
    ax.set_xlabel("Transverse Coordinate y (m)")
    ax.set_ylabel("Temperature T")
    ax.set_title(f"Temperature Profile Across Interface at x = {x_pos:.1f} m (Slope Jump: ks/kf = 10)")
    ax.legend(frameon=True)
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Interface slope jump plot saved to {save_path}")


def plot_xpinn_vs_monolithic(
    epochs: list,
    loss_xpinn: list,
    loss_monolithic: list,
    save_path: str = "figures/xpinn_vs_monolithic_comparison.png",
):
    set_publication_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.semilogy(epochs, loss_xpinn, "r-", lw=2, label="XPINN Dual-Subnet Loss")
    ax.semilogy(epochs, loss_monolithic, "b--", lw=2, label="Monolithic PINN Loss")

    ax.set_xlabel("Optimization Epoch")
    ax.set_ylabel("Total Training Loss")
    ax.set_title("Optimization Convergence: XPINN vs Monolithic PINN")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Convergence comparison saved to {save_path}")


def plot_nusselt_number(
    x_coords: np.ndarray,
    nu_exact: np.ndarray,
    nu_xpinn: np.ndarray,
    save_path: str = "figures/nusselt_number_distribution.png",
):
    set_publication_style()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x_coords, nu_exact, "k-", lw=2, label="Thermal Boundary Layer Benchmark")
    ax.plot(x_coords, nu_xpinn, "ro--", markersize=4, lw=1.5, label="XPINN Predicted Nu(x)")

    ax.set_xlabel("Axial Channel Position x / L")
    ax.set_ylabel("Local Nusselt Number Nu(x)")
    ax.set_title("Turbine Internal Cooling Passage Local Heat Transfer Rate")
    ax.legend()
    ax.grid(True, linestyle=":", alpha=0.6)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Nusselt number plot saved to {save_path}")
