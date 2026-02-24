import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

class PDEVisualizer:
    def __init__(self, grid):
        self.grid = grid
        self.S = grid.S

    def plot_value_vs_payoff(self, V, K, title="American Put Value"):
        """Plots the final option value against the intrinsic payoff."""
        payoff = np.maximum(K - self.S, 0)
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.S, V, label='Option Value (Numerical)', lw=2)
        plt.plot(self.S, payoff, 'r--', label='Intrinsic Value (Payoff)', alpha=0.7)
        
        # Highlight the free boundary S*(t)
        # Using a tolerance check to find where Value meets Payoff
        idx = np.where(np.isclose(V, payoff, atol=1e-4))[0]
        if len(idx) > 0:
            s_star = self.S[idx[-1]]
            plt.axvline(s_star, color='green', linestyle=':', label=f'Boundary S* ≈ {s_star:.2f}')
            
        plt.xlabel("Stock Price (S)")
        plt.ylabel("Option Price (V)")
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def plot_free_boundary(self, time_steps, boundary_history, title="Early Exercise Frontier"):
        """
        Deliverable: Free-boundary plot tracking S*(t) over time.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(time_steps, boundary_history, color='darkgreen', lw=2.5)
        
        plt.title(title)
        plt.xlabel("Time to Maturity (τ)")
        plt.ylabel("Optimal Exercise Boundary (S*)")
        plt.gca().invert_xaxis()  # Maturity (0) is on the right, T is on the left
        plt.grid(True, linestyle='--')
        plt.show()

    def plot_greeks(self, delta, gamma):
        """Visualizes the Greeks calculated in greeks.py."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(self.S, delta, color='blue', lw=1.5)
        ax1.set_title("Delta (Δ)")
        ax1.set_xlabel("S")
        ax1.grid(True, alpha=0.2)
        
        ax2.plot(self.S, gamma, color='purple', lw=1.5)
        ax2.set_title("Gamma (Γ)")
        ax2.set_xlabel("S")
        ax2.grid(True, alpha=0.2)
        
        plt.show()
    
    def plot_3d_surface(self, time_history, value_history, title="Option Value Surface"):
        """
        Plots V(S, t) as a 3D surface to show time-decay and boundary evolution.
        """
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # meshgrid(X, Y) where X is time and Y is Stock Price
        T_grid, S_grid = np.meshgrid(time_history, self.S)
        
        # We transpose if necessary to align with (S_grid, T_grid)
        surf = ax.plot_surface(S_grid, T_grid, value_history.T, cmap=cm.viridis, 
                               linewidth=0, antialiased=True, alpha=0.9)
        
        ax.set_xlabel('Stock Price (S)')
        ax.set_ylabel('Time to Maturity (τ)')
        ax.set_zlabel('Option Value (V)')
        ax.set_title(title)
        ax.view_init(elev=30, azim=-120)
        
        fig.colorbar(surf, shrink=0.5, aspect=5)
        plt.show()