import numpy as np

class PDEGrid:
    def __init__(self, S0, K, T, N_s=200, N_t=500, uniform=True):
        """
        S0: Current market spot price
        K: Strike price
        T: Time to maturity (years)
        N_s: Number of space steps
        N_t: Number of time steps
        uniform: Set to False for space refinement (denser nodes at lower prices)
        """
        self.S0 = S0
        self.K = K
        self.T = T
        self.N_s = N_s
        self.N_t = N_t
        self.dt = T / N_t
        
        # 1. Spatial Grid Setup
        # S_max needs to be large enough to represent 'infinity'
        # 2x the max(S0, K) is here to prevent boundary interference
        self.S_max = max(S0, self.K) * 2.0 
                
        if uniform:
            self.S = np.linspace(0, self.S_max, N_s + 1)
        else:
            # Space Refinement: geomspace places more nodes at the lower end
            # This helps track the American Put free boundary which is always < K
            # We start at 1e-5 to avoid log(0) issues.
            self.S = np.geomspace(1e-5, self.S_max, N_s + 1)
            self.S[0] = 0 # Force the first node to be exactly 0
        
        # 2. Pre-calculate Delta S (Array of distances between nodes)
        self.dS = np.diff(self.S)
        
        # 3. Time Grid
        # Working from tau = 0 (maturity) to tau = T (today)
        self.t_steps = np.linspace(0, T, N_t + 1)

    def get_indices_near_spot(self, S0_target=None):
        """
        Returns the index in the grid closest to a specific price.
        Defaults to the S0 provided at initialization.
        """
        target = S0_target if S0_target is not None else self.S0
        return np.argmin(np.abs(self.S - target))

    def get_payoff(self, option_type='put'):
        """Generates the terminal value vector (Payoff at T)."""
        if option_type == 'put':
            return np.maximum(self.K - self.S, 0)
        return np.maximum(self.S - self.K, 0)