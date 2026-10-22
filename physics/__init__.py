from physics.cht_equations import (
    fluid_energy_residual,
    solid_conduction_residual,
    interface_flux_residuals,
)
from physics.analytical_benchmarks import solve_conjugate_ground_truth

__all__ = [
    "fluid_energy_residual",
    "solid_conduction_residual",
    "interface_flux_residuals",
    "solve_conjugate_ground_truth",
]
