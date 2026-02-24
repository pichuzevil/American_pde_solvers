import time
import pandas as pd
import numpy as np
from data.market_loader import MarketDataLoader
from core.grid import PDEGrid
from core.pde_model import BlackScholesModel
from solvers.psor import PSORSolver
from solvers.penalty import PenaltySolver
from analysis.greeks import GreeksAnalyst
from analysis.visualizer import PDEVisualizer
from benchmark import run_benchmarks

def run_project():
    # DATA ACQUISITION
    ticker = "AAPL"
    print(f"Fetching market data for {ticker}...")
    loader = MarketDataLoader(ticker)
    
    # target_expiry_idx=2 typically picks a 3-month expiry
    params = loader.get_option_parameters(target_expiry_idx=2)
    
    print(f"Spot: {params['S0']:.2f} | Strike: {params['K']} | Vol: {params['sigma']:.2%}")
    print(f"Risk-free rate: {params['r']:.2%} | Dividend: {params['q']:.2%}")

    # GRID & MODEL SETUP
    # N_s=250 provides high resolution for the smooth-pasting check
    grid = PDEGrid(
        S0=params['S0'], 
        K=params['K'], 
        T=params['T'], 
        N_s=250, 
        N_t=500,
        uniform=False # Triggers non-uniform space refinement
    )
    
    model = BlackScholesModel(params)
    visualizer = PDEVisualizer(grid)
    analyst = GreeksAnalyst(grid)

    # SOLVER COMPARISON
    solvers = {
        "PSOR": PSORSolver(omega=1.2, tol=1e-7),
        "Penalty": PenaltySolver(rho=1e10, tol=1e-7)
    }
    
    results = {}
    performance_data = []

    for name, solver in solvers.items():
        print(f"Running {name} Solver...")
        start_time = time.time()
        
        # solver.solve() returns final V and internal state stores boundary history
        V_final = solver.solve(model, grid)
        
        duration = time.time() - start_time
        results[name] = V_final
        
        # Validation: Smooth-Pasting Check at current state
        s_star, delta_star, err = analyst.validate_smooth_pasting(V_final, params['K'])
        
        # Extract model price at current spot S0
        s0_idx = grid.get_indices_near_spot(params['S0'])
        model_price = V_final[s0_idx]
        market_diff = model_price - params['market_price']
        
        performance_data.append({
            "Method": name,
            "Model Price": f"{model_price:.4f}",
            "Market Price": f"{params['market_price']:.4f}",
            "Diff": f"{market_diff:.4f}",
            "Runtime (s)": f"{duration:.4f}",
            "Boundary S*": f"{s_star:.2f}",
            "Delta at S*": f"{delta_star:.4f}",
            "Pasting Error": f"{err:.6f}"
        })

    # REPORTING & VISUALIZATION
    print("\n--- Project Performance Comparison ---")
    df = pd.DataFrame(performance_data)
    print(df.to_string(index=False))

    # Choose the most accurate solver (Penalty) for detailed visuals
    main_v = results["Penalty"]
    main_solver = solvers["Penalty"]

    # Visual 1: Final Value vs Payoff & Smooth Pasting
    visualizer.plot_value_vs_payoff(main_v, params['K'], title=f"American Put Value ({ticker})")
    
    # Visual 2: The Free-Boundary Frontier S*(t)
    # This addresses the project requirement for free-boundary plots
    if hasattr(main_solver, 'boundary_history') and len(main_solver.boundary_history) > 0:
        # We plot against time-to-maturity (tau)
        tau_steps = np.linspace(0, params['T'], len(main_solver.boundary_history))
        visualizer.plot_free_boundary(tau_steps, main_solver.boundary_history)
        
    # Visual 3: Greeks (Delta and Gamma)
    delta = analyst.calculate_delta(main_v)
    gamma = analyst.calculate_gamma(main_v)
    visualizer.plot_greeks(delta, gamma)
    
    df_bench = run_benchmarks(params)
    print(df_bench.pivot(index='Nodes (Ns)', columns='Method', values=['Runtime (ms)', 'Pasting Error']))
    
if __name__ == "__main__":
    run_project()