import numpy as np
from scipy.stats import norm
from pricing.base import PricingEngine, Option


class BlackScholes(PricingEngine):
  """Analytical solution of the European option"""
  def calculate(self, option: Option, **kwargs) -> float:
    S, K, T, r, sigma, dividend, option_type, exercise_type = option.get_parameters()
    d1 = (np.log(S/K) + (r - dividend + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
      V = S * np.exp(-dividend * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
      V = K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-dividend * T) * norm.cdf(-d1)

    return float(V)


if __name__ == "__main__":
  test = Option(S=100, K=105, T=0.25, r=0.04, sigma=0.3) # ans. 4.32
  test = BlackScholes().calculate(test)
  if abs(test - 4.32) < 1e-2: 
    print(f"Call Option Price: {test: .4f}, the calculation is correct.")
  else:
    print(f"Call Option Price: {test: .4f}, the calculation is incorrect.")