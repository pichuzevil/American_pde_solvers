import time
import pandas as pd
import numpy as np
from core.grid import PDEGrid
from core.pde_model import BlackScholesModel
from solvers.psor import PSORSolver
from solvers.penalty import PenaltySolver
from analysis.greeks import GreeksAnalyst

def run_benchmarks(params):
    # Test different spatial grid sizes
    grid_sizes = [100, 250, 500, 750, 1000]
    benchmark_results = []

    # Reference value: Use a very high-resolution run or Binomial Tree
    # For this project, we'll treat the Penalty N=1000 run as the "Truth" 
    print("Generating Benchmark Data...")

    for ns in grid_sizes:
        # Keep time steps (nt) proportional to spatial steps for stability
        nt = ns * 2 
        grid = PDEGrid(S0=params['S0'], K=params['K'], T=params['T'], N_s=ns, N_t=nt, uniform=False)
        model = BlackScholesModel(params)
        analyst = GreeksAnalyst(grid)

        solvers = {
            "PSOR": PSORSolver(omega=1.2, tol=1e-8),
            "Penalty": PenaltySolver(rho=1e10, tol=1e-8)
        }

        for name, solver in solvers.items():
            start = time.time()
            v_final = solver.solve(model, grid)
            elapsed = time.time() - start

            # Accuracy Metric: Check Smooth-Pasting at the boundary
            s_star, delta_star, pasting_err = analyst.validate_smooth_pasting(v_final, params['K'])

            benchmark_results.append({
                "Nodes (Ns)": ns,
                "Method": name,
                "Runtime (ms)": elapsed * 1000,
                "S*": s_star,
                "Pasting Error": pasting_err
            })
            print(f"Finished {name} at Ns={ns}")

    return pd.DataFrame(benchmark_results)

