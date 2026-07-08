import numpy as np
from pricing.base import PricingEngine, Option


class MonteCarlo(PricingEngine):
  def calculate(self, option: Option, N=20, num_paths=1000000, method="standard", **kwargs) -> float:
    S, K, T, r, sigma, dividend, option_type, exercise_type = option.get_parameters()
    dt = T / N

    if option.exercise_type == "american":
        raise NotImplementedError(
            "Monte Carlo currently supports European options only. "
            "Pricing American options typically require the Longstaff-Schwartz (LSM) algorithm, "
            "which is outside the scope of this project."
        )

    # Choose the simulation method
    method = str(method).strip().lower()
    if method == "standard":
      return self._run_standard(S, K, T, r, sigma, dividend, option_type, N, num_paths, dt)
    elif method == "antithetic":
      return self._run_antithetic(S, K, T, r, sigma, dividend, option_type, N, num_paths, dt)
    elif method == "importance":
      return self._run_importance(S, K, T, r, sigma, dividend, option_type, N, num_paths, dt)
    else:
      raise ValueError(f"Invalid method: '{method}'. Expected 'standard', 'antithetic', or 'importance'.")

  # Standard
  def _run_standard(self, S, K, T, r, sigma, dividend, option_type, N, num_paths, dt):
    """Standard Monte Carlo simulation using Geometric Brownian Motion."""

    S_t = np.full(num_paths, S, dtype=np.float64) # Store only the current stock prices
    drift = ((r - dividend) - 0.5 * sigma**2) * dt
    vol_sqrt_dt = sigma * np.sqrt(dt)

    for _ in range(N):
      # Generate random standard normal values
      Z = np.random.normal(0.0, 1.0, num_paths)
      S_t *= np.exp(drift + vol_sqrt_dt * Z)

    payoffs = (S_t - K) if option_type == "call" else (K - S_t)
    discounted_payoffs = np.exp(-r * T) * np.maximum(payoffs, 0.0)

    return self._format_results(discounted_payoffs, num_paths)

  # Antithetic Variables
  def _run_antithetic(self, S, K, T, r, sigma, dividend, option_type, N, num_paths, dt):
    """Monte Carlo with antithetic variates for variance reduction."""

    # Make sure the number of paths is even
    if num_paths % 2 != 0:
      num_paths += 1

    S_t = np.full(num_paths, S, dtype=np.float64) # Store only the current stock prices
    drift = ((r - dividend) - 0.5 * sigma**2) * dt
    vol_sqrt_dt = sigma * np.sqrt(dt)
    half_paths = num_paths // 2

    for _ in range(N):
      # Combine Z and -Z to create antithetic pairs
      Z = np.random.normal(0.0, 1.0, half_paths)
      S_t[:half_paths] *= np.exp(drift + vol_sqrt_dt * Z)
      S_t[half_paths:] *= np.exp(drift + vol_sqrt_dt * (-Z))

    payoffs = (S_t - K) if option_type == "call" else (K - S_t)
    discounted_payoffs = np.exp(-r * T) * np.maximum(payoffs, 0.0)
    paired_payoffs = (discounted_payoffs[:half_paths] + discounted_payoffs[half_paths:]) / 2.0 # Pairwise average

    return self._format_results(paired_payoffs, num_paths)

  # Importance Sampling
  def _run_importance(self, S, K, T, r, sigma, dividend, option_type, N, num_paths, dt):
    """
    Monte Carlo with importance sampling.
    The probability distribution is shifted so that more paths
    end near the strike price (K). A likelihood ratio is then applied
    to keep the estimator unbiased.
    """

    S_t = np.full(num_paths, S, dtype=np.float64)

    likelihood_ratio = np.ones(num_paths, dtype=np.float64) # Each path starts with weight 1.0
    theta = (np.log(K / S) / T) - (r - dividend) # Drift adjustment
    delta = (theta / sigma) * np.sqrt(dt) # From dW_t (N(0, 1)) to dW_t* (N(delta, 1))
    drift = ((r - dividend) + theta - 0.5 * sigma**2) * dt
    vol_sqrt_dt = sigma * np.sqrt(dt)

    for _ in range(N):
      Z = np.random.normal(0.0, 1.0, num_paths)
      Y = Z + delta
      S_t *= np.exp(drift + vol_sqrt_dt * Z)

      # Likelihood ratio = p(Z) / q(Z), where p(Z) ~ N(0, 1) and q(Z) ~ N(delta, 1)
      likelihood_ratio *= np.exp(-Y * delta + 0.5 * delta**2)

    payoffs = (S_t - K) if option_type == "call" else (K - S_t)
    discounted_payoffs = np.exp(-r * T) * np.maximum(payoffs, 0.0) * likelihood_ratio # Apply the likelihood ratio to correct the probability change

    return self._format_results(discounted_payoffs, num_paths)

  # Compute the result and summary statistics
  def _format_results(self, discounted_payoffs, num_paths) -> dict:
    V = np.mean(discounted_payoffs)
    sample_std = np.std(discounted_payoffs, ddof=1)
    standard_error = sample_std / np.sqrt(num_paths)

    return {
      "V": float(V),
      "standard_error": float(standard_error),
      "ci_95_lower": float(V - 1.96 * standard_error),
      "ci_95_upper": float(V + 1.96 * standard_error)
    }


if __name__ == "__main__":
  test = Option(S=100, K=105, T=0.25, r=0.04, sigma=0.3) # ans. 4.32
  test1 = MonteCarlo().calculate(test)
  test2 = MonteCarlo().calculate(test, method="antithetic")
  if abs(test1["V"] - 4.32) < 1e-2 and abs(test2["V"] - 4.32) < 1e-2:
    print(f"Call Option Price: {test1['V']: .4f} (standard) & {test2['V']: .4f} (antithetic variables),"
          " the calculation is close enough to the analytical solution.")
    print(f"Also, the standard confidence interval [{test1['ci_95_lower']: .4f}, {test1['ci_95_upper']: .4f} ] is wider"
          f" than that of the antithetic variables [{test2['ci_95_lower']: .4f}, {test2['ci_95_upper']: .4f} ]." )
  else:
    print(f"Call Option Price: {test1['V']: .4f} (standard) & {test2['V']: .4f} (antithetic variables), the calculation is incorrect.")