import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

class PenaltySolver:
    def __init__(self, tol=1e-7, max_iter=100, rho=1e10):
        self.tol = tol
        self.max_iter = max_iter
        self.rho = rho

    def solve(self, model, grid, option_type='put'):
        # Initialize full-sized vectors (size N+1)
        V = grid.get_payoff(option_type)
        payoff = V.copy()
        N = grid.N_s
        
        # Get PDE coefficients and build the (N-1)x(N-1) Matrix
        a, b, c = model.get_coefficients(grid)
        
        # Construct the Crank-Nicolson Operator (L) for interior nodes
        main_diag = 1 - b
        off_diag_low = -a[1:]   # Sliced to match (N-2) length for off-diagonals
        off_diag_high = -c[:-1] 
        
        L_matrix = sparse.diags([off_diag_low, main_diag, off_diag_high], [-1, 0, 1], format='csr')

        # Outer Time Loop
        for j in range(grid.N_t):
            # Calculate RHS for the full grid (N+1)
            rhs_full = model.calculate_rhs(V, grid, a, b, c)
            
            # Slice RHS and Payoff to only include INTERIOR nodes (1 to N-1)
            rhs_int = rhs_full[1:-1]
            pay_int = payoff[1:-1]
            v_int = V[1:-1] # Initial guess for Newton iteration is previous time step
            
            # 4. Inner Newton-style Penalty Iteration
            for _ in range(self.max_iter):
                V_old_int = v_int.copy()
                
                # Check which nodes need the penalty
                current_mask = (v_int < pay_int).astype(float)
                
                # Optimization : If the exercise boundary hasn't moved, we've converged for this step
                if _ > 0 and np.array_equal(current_mask, last_mask):
                    break
                last_mask = current_mask.copy()

                D = sparse.diags(current_mask * self.rho, format='csr')
                A_system = L_matrix + D
                b_system = rhs_int + D @ pay_int
                
                v_int = spsolve(A_system, b_system)
            
            # 5. Map solved interior back to the full V vector
            V[1:-1] = v_int
            
            # Explicitly maintain Boundary Conditions
            V[0] = payoff[0] 
            V[-1] = payoff[-1]
                    
        return V