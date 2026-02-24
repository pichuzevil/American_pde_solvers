import numpy as np

class BlackScholesModel:
    def __init__(self, params):
        self.r = params['r']
        # Ensure sigma is at least a small number to avoid division by zero/kinks
        self.sigma = max(params['sigma'], 1e-4) 
        self.q = params.get('q', 0.0)

    def get_coefficients(self, grid):
        S = grid.S
        dt = grid.dt
        N = grid.N_s
        
        a, b, c = np.zeros(N - 1), np.zeros(N - 1), np.zeros(N - 1)
        
        for i in range(1, N):
            idx = i - 1
            # Local grid spacings for non-uniform grid refinement
            h_prev = S[i] - S[i-1]
            h_next = S[i+1] - S[i]
            h_sum = h_prev + h_next
            
            # Black-Scholes operator components
            drift = (self.r - self.q) * S[i]
            diff = 0.5 * (self.sigma**2) * (S[i]**2)
            
            # Non-uniform finite difference stencil
            L_prev = (2 * diff / (h_prev * h_sum)) - (drift / h_sum)
            L_curr = - (2 * diff / (h_prev * h_next)) - self.r
            L_next = (2 * diff / (h_next * h_sum)) + (drift / h_sum)
            
            # Crank-Nicolson implicit multipliers
            a[idx] = 0.5 * dt * L_prev
            b[idx] = 0.5 * dt * L_curr
            c[idx] = 0.5 * dt * L_next
            
        return a, b, c

    def calculate_rhs(self, V, grid, a, b, c):
        rhs = np.zeros_like(V)
        N = grid.N_s
        # Explicit half of Crank-Nicolson
        for i in range(1, N):
            idx = i - 1
            rhs[i] = a[idx]*V[i-1] + (1 + b[idx])*V[i] + c[idx]*V[i+1]
            
        # Boundary conditions for american put
        rhs[0] = grid.K 
        rhs[N] = 0.0    
        return rhs