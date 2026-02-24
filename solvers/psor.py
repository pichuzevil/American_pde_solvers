import numpy as np

class PSORSolver:
    def __init__(self, omega=1.2, tol=1e-7, max_iter=1000):
        """
        omega: Relaxation parameter (typically 1.0 < omega < 2.0)
        tol: Convergence tolerance for the inner loop
        max_iter: Maximum number of iterations per time step
        """
        self.omega = omega
        self.tol = tol
        self.max_iter = max_iter

    def solve(self, model, grid, option_type='put'):
        """
        model: BlackScholesModel object
        grid: PDEGrid object
        """
        # Initialize values and pre-calculate coefficients
        V = grid.get_payoff(option_type)
        payoff = V.copy()
        a, b, c = model.get_coefficients(grid)
        
        # Outer time loop (Backward in time)
        for j in range(grid.N_t):
            # Calculate the explicit RHS for the current time step
            rhs = model.calculate_rhs(V, grid, a, b, c)
            
            # 3. Inner PSOR loop (Iterative LCP Solver)
            for it in range(self.max_iter):
                V_old = V.copy()
                
                # Update interior nodes
                for i in range(1, grid.N_s):
                    idx = i - 1
                    
                    # Gauss-Seidel isolation of V[i]
                    z = (rhs[i] + a[idx]*V[i-1] + c[idx]*V[i+1]) / (1 - b[idx])
                    
                    # SOR Update with Projection (American Constraint)
                    # V[i] = max( Relaxation, Payoff )
                    new_val = (1 - self.omega) * V[i] + self.omega * z
                    V[i] = max(new_val, payoff[i])
                
                # Check for convergence within the time step
                if np.linalg.norm(V - V_old, ord=np.inf) < self.tol:
                    break
                    
        return V