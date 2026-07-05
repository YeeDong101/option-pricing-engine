# pricing/binomial_tree.py
import numpy as np
from scipy.stats import binom
from pricing.base import PricingEngine, Option


class BinomialTree(PricingEngine):
  def calculate(self, option: Option, N=1000, **kwargs) -> float:
    S, K, T, r, sigma, dividend, option_type, exercise_type = option.get_parameters()
    dt = T / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp((r - dividend) * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    j = np.arange(N + 1) # Numbers of price going up
    S_T = S * ((u ** j) * (d ** (N - j))) # Final value of stock price

    # Initialize the payoffs
    if option_type == "call":
        V = np.maximum(S_T - K, 0.0)
    else:
        V = np.maximum(K - S_T, 0.0)

    # European options
    if exercise_type == "european":
      # Calculate binomial probabilities and expected payoff
      probabilities = binom.pmf(j, N, p)
      expected_payoff = np.sum(V * probabilities)
      V_0 = expected_payoff * np.exp(-r * T) # Discount expected value back to present
      
      return float(V_0)

    # American options
    else: 
      # Backward Induction (from step N-1)
      for i in range(N - 1, -1, -1):
        j_i = np.arange(i + 1)
        S_i = S * ((u ** j_i) * (d ** (i - j_i)))

        # Vectorized Operation
        V_i = discount * (p * V[1:] + (1-p) * V[:-1]) # Price goes up (V[1:]) and down (V[:-1]) at time i+1

        # Continue to hold or early exercise
        if option_type == "call":
          V = np.maximum(V_i, np.maximum(S_i - K, 0))
        else:
          V = np.maximum(V_i, np.maximum(K - S_i, 0))

      return float(V[0])


if __name__ == "__main__":
  test = Option(S=100, K=105, T=0.25, r=0.04, sigma=0.3) # ans. 4.32
  test = BinomialTree().calculate(test, 100000)
  if abs(test - 4.32) < 1e-2: 
    print(f"Call Option Price: {test: .4f}, the calculation is close enough to the analytical solution.")
  else:
    print(f"Call Option Price: {test: .4f}, the calculation is incorrect.")