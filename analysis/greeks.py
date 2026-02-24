import numpy as np

class GreeksAnalyst:
    def __init__(self, grid):
        self.grid = grid
        self.S = grid.S
        self.ds = grid.dS # To handle non-uniform spacing

    def calculate_delta(self, V):
        """Calculates Delta (dV/dS) using central differences."""
        delta = np.zeros_like(V)
        # For non-uniform grids, we use the generalized central difference
        for i in range(1, len(V) - 1):
            h_prev = self.S[i] - self.S[i-1]
            h_next = self.S[i+1] - self.S[i]
            # Weighted central difference
            delta[i] = (V[i+1] - V[i-1]) / (h_prev + h_next)
        
        # Boundary handling (one-sided differences)
        delta[0] = (V[1] - V[0]) / (self.S[1] - self.S[0])
        delta[-1] = (V[-1] - V[-2]) / (self.S[-1] - self.S[-2])
        return delta

    def calculate_gamma(self, V):
        """Calculates Gamma (d^2V/dS^2) for the price surface."""
        gamma = np.zeros_like(V)
        for i in range(1, len(V) - 1):
            h_prev = self.S[i] - self.S[i-1]
            h_next = self.S[i+1] - self.S[i]
            # Non-uniform second derivative formula
            gamma[i] = 2 * ((V[i+1] - V[i]) / h_next - (V[i] - V[i-1]) / h_prev) / (h_next + h_prev)
        return gamma

    def find_free_boundary(self, V, K):
        """
        Deliverable: Find S*(t) where the option value hits the payoff.
        For a Put: The largest S where V(S) is approximately K - S.
        """
        payoff = np.maximum(K - self.S, 0)
        # Only look for the boundary where the payoff is actually positive (S < K)
        exercise_nodes = np.where((np.abs(V - payoff) < 1e-4) & (self.S < K))[0]
        
        if len(exercise_nodes) > 0:
            s_star_idx = exercise_nodes[-1]
            return self.S[s_star_idx], s_star_idx
        return None, None

    def validate_smooth_pasting(self, V, K):
        """
        Deliverable: Checks if Delta = -1 at the exercise boundary.
        """
        s_star, idx = self.find_free_boundary(V, K)
        if idx is not None:
            delta = self.calculate_delta(V)
            boundary_delta = delta[idx]
            error = abs(boundary_delta - (-1.0))
            return s_star, boundary_delta, error
        return None, None, None