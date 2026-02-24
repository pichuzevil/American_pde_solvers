# American Option Pricing: PDE Numerical Methods

This repository contains a high-performance Python implementation for pricing **American Put Options** using Finite Difference Methods (FDM). It explores two primary numerical engines for solving the Linear Complementarity Problem (LCP) associated with early exercise: the **Projected Successive Over-Relaxation (PSOR)** method and the **Penalty Method** (Newton-type solver).

## Features

* **Dual Numerical Engines**: Implementation of both PSOR and Penalty solvers for comparative analysis.
* **Space Refinement**: Non-uniform grid construction using log-spacing to increase accuracy near the early-exercise boundary.
* **Real-Market Integration**: Automated data fetching for stock prices (AAPL), implied volatility, and risk-free rates via `yfinance`.
* **Greeks & Validation**: Numerical calculation of Delta and Gamma with specific validation of the **Smooth-Pasting Condition** ( at the boundary).
* **Visualization Suite**: 2D/3D plots of the option value surface and the dynamic Early Exercise Frontier .

---

## Project Structure

```text
american_pde_project/
│
├── core/
│   ├── grid.py           # Non-uniform spatial and temporal grid generation
│   └── pde_model.py      # Black-Scholes operator with variable coefficients
│
├── solvers/
│   ├── psor.py           # PSOR iterative solver for LCP
│   └── penalty.py        # Newton-type Penalty solver using scipy.sparse
│
├── analysis/
│   ├── greeks.py         # Finite difference Greeks and pasting validation
│   └── visualizer.py     # Matplotlib 2D/3D visualization suite
│
├── data/
│   └── market_loader.py  # Market data acquisition and volatility fallbacks
│
└── main.py               # Orchestration and benchmarking entry point

```

---

## Numerical Results & Benchmarks

The project includes a comprehensive benchmarking suite that compares the solvers across various grid densities.

### Performance Summary (AAPL Example)

| Method | Price | Runtime (s) | Boundary  | Delta at  | Pasting Error |
| --- | --- | --- | --- | --- | --- |
| **PSOR** | 7.7436 | 1.5404 | 233.79 | -0.9963 | 0.003687 |
| **Penalty** | 7.7436 | 23.2930 | 233.79 | -0.9963 | 0.003687 |

### Convergence Table

Increasing the number of nodes () significantly reduces the "Pasting Error," validating the second-order accuracy of the non-uniform scheme.

| Nodes () | PSOR Runtime (ms) | Penalty Runtime (ms) | Pasting Error |
| --- | --- | --- | --- |
| 100 | 552.50 | 148.54 | 0.0158 |
| 500 | 7366.83 | 56260.71 | 0.0045 |
| 1000 | 27743.28 | 212659.95 | 0.00005 |

---

## Installation & Usage

### Prerequisites

* Python 3.10+
* `numpy`, `scipy`, `pandas`, `matplotlib`, `yfinance`

### Run the Project

```bash
git clone https://github.com/yourusername/american-option-pde.git
cd american-option-pde
python main.py

```

---

## Key Mathematical Concepts

### The Smooth-Pasting Condition

Unlike European options, American Put values must merge tangentially into the payoff function at the optimal exercise boundary . This repository validates that:
$$\frac{\partial V}{\partial S} \bigg|_{S = S^*} = -1$$ 


### Non-Uniform Finite Difference

To handle the "space refinement" requirement, we utilize a generalized stencil for the second derivative:

$$\frac{\partial^2 V}{\partial S^2} \approx \frac{2}{h_i + h_{i-1}} \left( \frac{V_{i+1} - V_i}{h_i} - \frac{V_i - V_{i-1}}{h_{i-1}} \right)$$

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.