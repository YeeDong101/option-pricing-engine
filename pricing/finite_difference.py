# pricing/finite_difference.py
import numpy as np
from pricing.base import PricingEngine, Option


class FiniteDifference(PricingEngine):
  """Implicit finite difference (M: price grid size, N: time grid size, max_ratio: upper bound multiplier relative to strike price)."""
  def calculate(self, option: Option, M=1000, N=2000, max_ratio=3, return_grid=False, **kwargs) -> float:
    S, K, T, r, sigma, dividend, option_type, exercise_type = option.get_parameters()
    dt = T / N
    S_max = max_ratio * K

    # Adjust grid points dynamically so that S lands exactly on a node
    i_S = int(round(M * S / S_max))
    dS = S / i_S
    M = int(np.ceil(S_max / dS))
    S_max = M * dS # New upper bound

    # Price grid points: 0, dS, 2dS, ..., M*dS
    S_space = np.linspace(0, S_max, M + 1)

    # Initialize option prices & the Greeks
    V_t = np.zeros(M + 1)
    if option_type == "call":
        payoffs = np.maximum(S_space - K, 0.0)
    else:
        payoffs = np.maximum(K - S_space, 0.0)
    V_t[:] = payoffs # Option prices at time t

    if return_grid:
      V_grid = np.zeros((N + 1, M + 1)) # Option prices of each grid
      V_grid[-1, :] = payoffs
      delta_grid = np.zeros((N + 1, M + 1)) # Delta
      gamma_grid = np.zeros((N + 1, M + 1)) # Gamma
      theta_grid = np.zeros((N + 1, M + 1)) # Theta

    # Precompute tridiagonal matrix A coefficients (a, b, c) from discretized BS PDE,
    # substitute S = j * dS eliminates dS; coefficients depend only on integer j.
    j = np.arange(1, M) # indices of internal nodes from 1 to M-1
    a = 0.5 * (r - dividend) * j * dt - 0.5 * (sigma**2) * (j**2) * dt
    b = 1.0 + (sigma**2) * (j**2) * dt + r * dt
    c = -0.5 * (r - dividend) * j * dt - 0.5 * (sigma**2) * (j**2) * dt

    # Backward Induction (from t = T-dt to t = 0)
    for i in range(N - 1, -1, -1):
        t_current = i * dt

        if i == 0:
          V_1 = V_t.copy() # For theta calculations

        # Calculate dynamic boundary conditions at both ends (S=0 and S=S_max)
        if option_type == "call":
            V_S0 = 0.0
            V_Smax = S_max * np.exp(-dividend * (T - t_current)) - K * np.exp(-r * (T - t_current))
        else:
            V_S0 = K * np.exp(-r * (T - t_current))
            V_Smax = 0.0

        # Construct RHS vector 'd' for Ax = d using future option values
        d = V_t[1:M].copy()

        # Adjust RHS vector for boundary conditions at first and last internal nodes
        d[0] = d[0] - a[0] * V_S0       # a[0] corresponds to index j=1
        d[-1] = d[-1] - c[-1] * V_Smax   # c[-1] corresponds to index j=M-1

        # Solve the tridiagonal system for current internal node values
        V_next = self.thomas_algorithm(a, b, c, d)

        # Check early exercise for American options
        if exercise_type.lower() == "american":
            V_t[1:M] = np.maximum(V_next, payoffs[1:M])
        else:
            V_t[1:M] = V_next

        # Enforce boundary values explicitly on the grid edges
        V_t[0] = V_S0
        V_t[M] = V_Smax

        if return_grid:
          V_grid[i, :] = V_t # Store the prices every time

    if return_grid:
      # Central Difference to calculate the Greeks
      delta_grid[:, 1:M] = (V_grid[:, 2:M + 1] - V_grid[:, 0:M - 1]) / (2 * dS)
      gamma_grid[:, 1:M] = (V_grid[:, 2:M + 1] - 2 * V_grid[:, 1:M] + V_grid[:, 0:M - 1]) / (dS ** 2)
      theta_grid[0:N, :] = (V_grid[1:N + 1, :] - V_grid[0:N, :]) / dt

      return {
        "V": float(V_grid[0, i_S]),
        "delta": float(delta_grid[0, i_S]),
        "gamma": float(gamma_grid[0, i_S]),
        "theta_per_year": float(theta_grid[0, i_S]),
        "theta_per_day": float(theta_grid[0, i_S] / 365.0),
        "V_grid": V_grid,
        "delta_grid": delta_grid,
        "gamma_grid": gamma_grid,
        "theta_grid": theta_grid,
        "S_space": S_space,
        "dt": dt,
      }

    else:
      V = V_t[i_S]

      # Central Difference to calculate the Greeks
      delta = (V_t[i_S + 1] - V_t[i_S - 1]) / (2 * dS)
      gamma = (V_t[i_S + 1] - 2 * V_t[i_S] + V_t[i_S - 1]) / (dS ** 2)
      theta = (V_1[i_S] - V_t[i_S]) / dt

      return {
        "V": float(V),
        "delta": float(delta),
        "gamma": float(gamma),
        "theta_per_year": float(theta),
        "theta_per_day": float(theta / 365.0)
      }

  def thomas_algorithm(self, a, b, c, d):
    """
    Thomas Algorithm (Tridiagonal Matrix Solver).
    Efficiently solves the linear system Ax = d with a time complexity of O(M).
    Args:
        a (1D array): Sub-diagonal elements.
        b (1D array): Main diagonal elements.
        c (1D array): Super-diagonal elements.
        d (1D array): Right-hand side (RHS) vector.
    Returns:
        1D array: Solution vector x.
    """
    n = len(d)
    c_prime = np.zeros(n)
    d_prime = np.zeros(n)
    x = np.zeros(n)

    # 1. Forward Sweep
    c_prime[0] = c[0] / b[0]
    d_prime[0] = d[0] / b[0]

    for i in range(1, n):
        denominator = b[i] - a[i] * c_prime[i-1]
        if i < n - 1:
            c_prime[i] = c[i] / denominator
        d_prime[i] = (d[i] - a[i] * d_prime[i-1]) / denominator

    # 2. Backward Substitution
    x[-1] = d_prime[-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]

    return x


if __name__ == "__main__":
  test = Option(S=100, K=105, T=0.25, r=0.04, sigma=0.3) # ans. 4.32
  test = FiniteDifference().calculate(test)
  if abs(test["V"] - 4.32) < 1e-2:
    print(f"Call Option Price: {test['V']: .4f}, the calculation is close enough to the analytical solution.")
    print(f"Also, the estimated Greeks are Delta = {test['delta']: .4f}, Gamma = {test['gamma']: .4f},"
          f" and Theta per year (per day) = {test['theta_per_year']: .4f} ({test['theta_per_day']: .4f}).")
  else:
    print(f"Call Option Price: {test['V']: .4f}, the calculation is incorrect.")