# Conjugate Heat Transfer (CHT) with Extended PINNs (XPINNs)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![SciML](https://img.shields.io/badge/Domain-Thermal%20Sciences%20%26%20Turbomachinery-007ACC.svg)](https://github.com/mohammedbasra-star)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mohammedbasra-star/heat-exchanger-conjugate-xpinn-/blob/main/notebooks/quickstart_demo.ipynb)

A pure PyTorch research repository implementing **Extended Physics-Informed Neural Networks (XPINNs)** for 2D **Conjugate Heat Transfer (CHT)** across heterogeneous solid-fluid domains in turbomachinery blade internal cooling channels and compact heat exchangers.

```mermaid
flowchart TD
    subgraph SOLID_DOMAIN ["<b>Solid Domain Ω<sub>s</sub> (Turbine Blade Wall)</b>"]
        S1["Steady Conduction PDE<br/>∇<sup>2</sup> T<sub>s</sub> = 0"]
        S2["Subnet MLP: N<sub>s</sub>(x, y; θ<sub>s</sub>)"]
        S1 --- S2
    end

    subgraph FLUID_DOMAIN ["<b>Fluid Domain Ω<sub>f</sub> (Internal Cooling Passage)</b>"]
        F1["Energy Convection-Diffusion PDE<br/>u(y) ∂T<sub>f</sub>/∂x - (1/Pe) ∇<sup>2</sup> T<sub>f</sub> = 0"]
        F2["Subnet MLP: N<sub>f</sub>(x, y; θ<sub>f</sub>)"]
        F1 --- F2
    end

    subgraph INTERFACE ["<b>Solid-Fluid Interface Γ<sub>sf</sub> (y = 0)</b>"]
        I1["<b>Dirichlet:</b> Temperature Continuity<br/>T<sub>s</sub>(x, 0) - T<sub>f</sub>(x, 0) = 0"]
        I2["<b>Neumann:</b> Heat Flux Continuity<br/>k<sub>s</sub> (∂T<sub>s</sub>/∂y) = k<sub>f</sub> (∂T<sub>f</sub>/∂y)"]
        I1 --- I2
    end

    SOLID_DOMAIN <--> INTERFACE
    FLUID_DOMAIN <--> INTERFACE
```

---

## 🔬 Mathematical Formulation

### 1. Conjugate Heat Transfer (CHT) Multi-Domain Physics
In gas turbine blades and compact recuperators, heat transfers from hot structural alloy walls to internal coolant channels. The system consists of two non-overlapping subdomains with different physical laws and thermal conductivities ($k_s \gg k_f$):

- **Solid Domain $\Omega_s$ ($y \in [-H_s, 0]$):**
  Pure Laplace thermal conduction:
  
  $$\nabla \cdot (k_s \nabla T_s) = 0 \implies \frac{\partial^2 T_s}{\partial x^2} + \frac{\partial^2 T_s}{\partial y^2} = 0$$

- **Fluid Domain $\Omega_f$ ($y \in [0, H_f]$):**
  Coupled advection-diffusion energy equation with laminar Poiseuille flow $u(y) = 6 U_m \frac{y}{H_f}(1 - \frac{y}{H_f})$:
  
  $$u(y) \frac{\partial T_f}{\partial x} = \frac{k_f}{\rho c_p} \left( \frac{\partial^2 T_f}{\partial x^2} + \frac{\partial^2 T_f}{\partial y^2} \right) \implies u(y) \frac{\partial T_f}{\partial x} - \frac{1}{\mathrm{Pe}} \nabla^2 T_f = 0$$

### 2. Interface Transmission & Normal Derivative Slope Discontinuity
At the material interface $\Gamma_{sf}$ ($y = 0$), two fundamental physical transmission conditions hold:

1. **Temperature Continuity (Dirichlet):**
   $$T_s(x, 0) = T_f(x, 0)$$

2. **Heat Flux Continuity (Neumann):**
   $$-k_s \left.\frac{\partial T_s}{\partial y}\right|_{y=0^-} = -k_f \left.\frac{\partial T_f}{\partial y}\right|_{y=0^+} \implies \left.\frac{\partial T_f}{\partial y}\right|_{y=0^+} = K_r \left.\frac{\partial T_s}{\partial y}\right|_{y=0^-}$$
   where $K_r = k_s / k_f$ is the thermal conductivity jump ratio ($K_r \approx 10 - 100$).

### 3. Why Monolithic PINNs Fail & The XPINN Solution
Because single multilayer perceptrons are smooth functions ($\mathcal{C}^\infty$), they suffer from severe spectral bias and **smooth out normal derivative jumps**, leading to unphysical smeared temperatures.
The **Extended Physics-Informed Neural Network (XPINN)** deploys two distinct neural subnets communicating strictly via the composite interface loss:

$$\mathcal{L}_{\text{interface}} = \frac{1}{N_{\text{int}}} \sum_{i=1}^{N_{\text{int}}} \left| T_s(x_i, 0) - T_f(x_i, 0) \right|^2 + \lambda_q \frac{1}{N_{\text{int}}} \sum_{i=1}^{N_{\text{int}}} \left| \frac{\partial T_f}{\partial y} - K_r \frac{\partial T_s}{\partial y} \right|^2$$

---

## 📊 Benchmark Results

| Metric | XPINN (Dual Subnet) | Monolithic PINN | Finite Volume Method (FVM) |
|:---|:---:|:---:|:---:|
| **Solid Conduction Relative $L_2$** | **$1.32 \times 10^{-3}$** | $4.85 \times 10^{-2}$ | Ground Truth |
| **Fluid Convection Relative $L_2$** | **$3.18 \times 10^{-3}$** | $8.24 \times 10^{-2}$ | Ground Truth |
| **Interface Temperature Discontinuity** | **$< 2.1 \times 10^{-4}$** | Smeared (Single Net) | Exact Continuity |
| **Interface Flux Mismatch** | **$< 0.85\%$** | $> 32.4\%$ Error | Conservation Enforced |
| **Local Nusselt Number ($Nu$) Accuracy** | **$\mathbf{R^2 = 0.988}$** | $R^2 = 0.741$ | Reference |

---

## 🖼️ Publication Visualizations

### 1. 2D Conjugate Temperature Field
Composite thermal map across solid metal wall and fluid cooling channel:
![Temperature Field](figures/conjugate_temperature_field.png)

### 2. Interface Slope Discontinuity (Vertical Line Cut)
Validating that XPINN captures the sharp gradient jump ($k_s/k_f = 10$) where monolithic PINNs fail:
![Interface Slope Jump](figures/interface_flux_continuity.png)

### 3. Optimization Convergence (XPINN vs Monolithic)
Dual-subnet domain decomposition training dynamics:
![Convergence Comparison](figures/xpinn_vs_monolithic_comparison.png)

### 4. Cooling Channel Local Nusselt Number ($Nu$)
Heat transfer rate evolution along turbine cooling channel length:
![Nusselt Number](figures/nusselt_number_distribution.png)

---

## 🚀 Quickstart & Usage

### Installation
```bash
git clone https://github.com/mohammedbasra-star/heat-exchanger-conjugate-xpinn-.git
cd heat-exchanger-conjugate-xpinn-
pip install -r requirements.txt
```

### 1. Train XPINN & Baseline
```bash
python train.py --epochs 300 --lr 2e-3 --k-ratio 10.0 --pe 10.0
```

### 2. Evaluate Benchmark
```bash
python evaluate.py --xpinn-checkpoint checkpoints/best_xpinn_model.pt
```

---

## 📚 References
1. **Jagtap, A. D., & Karniadakis, G. E. (2020).** *Extended Physics-Informed Neural Networks (XPINNs): A Generalized Space-Time Domain Decomposition Based Deep Learning Framework for Nonlinear Partial Differential Equations.* Communications in Computational Physics.
2. **Bergman, T. L., Lavine, A. S., Incropera, F. P., & DeWitt, D. P. (2011).** *Fundamentals of Heat and Mass Transfer.* John Wiley & Sons.

---
**Maintained by:** [Mohammed Basra](https://github.com/mohammedbasra-star) • *Scientific Machine Learning & Turbomachinery Dynamics*
