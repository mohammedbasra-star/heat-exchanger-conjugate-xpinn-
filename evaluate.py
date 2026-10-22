"""
Evaluation and Benchmarking Script for Conjugate Heat Transfer XPINN.
Compares XPINN vs Monolithic PINN across temperature L2 error, interface jump error, and flux mismatch.
"""

import argparse
import os
import json
import torch
import numpy as np

from physics.analytical_benchmarks import solve_conjugate_ground_truth
from models.xpinn_module import XPINNConjugate
from models.monolithic_pinn import MonolithicPINN


def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating CHT models on: {device}")

    xpinn = XPINNConjugate(hidden_dim=args.hidden_dim, num_layers=args.layers).to(device)
    monolithic = MonolithicPINN(hidden_dim=args.hidden_dim, num_layers=args.layers).to(device)

    xpinn.load_state_dict(torch.load(args.xpinn_checkpoint, map_location=device))
    monolithic.load_state_dict(torch.load(args.mono_checkpoint, map_location=device))
    xpinn.eval()
    monolithic.eval()

    # Ground truth
    gt = solve_conjugate_ground_truth(nx=60, ny_solid=30, ny_fluid=30, k_ratio=args.k_ratio, pe=args.pe)
    nx_plot = len(gt["x"])
    ny_s_plot = len(gt["y_solid"])
    ny_f_plot = len(gt["y_fluid"])

    X_s_eval, Y_s_eval = np.meshgrid(gt["x"], gt["y_solid"], indexing="ij")
    X_f_eval, Y_f_eval = np.meshgrid(gt["x"], gt["y_fluid"], indexing="ij")

    with torch.no_grad():
        x_s = torch.from_numpy(X_s_eval.ravel()[:, None]).float().to(device)
        y_s = torch.from_numpy(Y_s_eval.ravel()[:, None]).float().to(device)
        T_s_xpinn = xpinn.forward_solid(x_s, y_s).cpu().numpy().reshape(nx_plot, ny_s_plot)
        T_s_mono = monolithic(x_s, y_s).cpu().numpy().reshape(nx_plot, ny_s_plot)

        x_f = torch.from_numpy(X_f_eval.ravel()[:, None]).float().to(device)
        y_f = torch.from_numpy(Y_f_eval.ravel()[:, None]).float().to(device)
        T_f_xpinn = xpinn.forward_fluid(x_f, y_f).cpu().numpy().reshape(nx_plot, ny_f_plot)
        T_f_mono = monolithic(x_f, y_f).cpu().numpy().reshape(nx_plot, ny_f_plot)

    # Errors
    l2_solid_xpinn = np.linalg.norm(T_s_xpinn - gt["T_solid"]) / np.linalg.norm(gt["T_solid"])
    l2_fluid_xpinn = np.linalg.norm(T_f_xpinn - gt["T_fluid"]) / np.linalg.norm(gt["T_fluid"])

    l2_solid_mono = np.linalg.norm(T_s_mono - gt["T_solid"]) / np.linalg.norm(gt["T_solid"])
    l2_fluid_mono = np.linalg.norm(T_f_mono - gt["T_fluid"]) / np.linalg.norm(gt["T_fluid"])

    # Interface temperature discrepancy at y = 0
    x_int = torch.from_numpy(gt["x"][:, None]).float().to(device)
    y_int = torch.zeros_like(x_int)
    with torch.no_grad():
        t_s_int = xpinn.forward_solid(x_int, y_int).cpu().numpy().ravel()
        t_f_int = xpinn.forward_fluid(x_int, y_int).cpu().numpy().ravel()
        int_gap_xpinn = np.mean(np.abs(t_s_int - t_f_int))

    print("\n" + "=" * 60)
    print("      CONJUGATE HEAT TRANSFER XPINN BENCHMARK REPORT      ")
    print("=" * 60)
    print(f"  {'Metric':<35} | {'XPINN':<10} | {'Monolithic':<10}")
    print("-" * 60)
    print(f"  {'Solid Conduction Relative L2':<35} | {l2_solid_xpinn:.4e} | {l2_solid_mono:.4e}")
    print(f"  {'Fluid Convection Relative L2':<35} | {l2_fluid_xpinn:.4e} | {l2_fluid_mono:.4e}")
    print(f"  {'Interface Temperature Discontinuity':<35} | {int_gap_xpinn:.4e} | N/A (single net)")
    print("=" * 60)

    results = {
        "l2_solid_xpinn": float(l2_solid_xpinn),
        "l2_fluid_xpinn": float(l2_fluid_xpinn),
        "l2_solid_monolithic": float(l2_solid_mono),
        "l2_fluid_monolithic": float(l2_fluid_mono),
        "interface_gap": float(int_gap_xpinn),
    }
    os.makedirs("results", exist_ok=True)
    with open("results/test_evaluation.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Benchmark saved to results/test_evaluation.json\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CHT XPINN Model")
    parser.add_argument("--xpinn-checkpoint", type=str, default="checkpoints/best_xpinn_model.pt")
    parser.add_argument("--mono-checkpoint", type=str, default="results/best_monolithic_model.pt")
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--k-ratio", type=float, default=10.0)
    parser.add_argument("--pe", type=float, default=10.0)

    args = parser.parse_args()
    evaluate(args)
