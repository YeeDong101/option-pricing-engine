from abc import ABC, abstractmethod


class Option:
  """Encapsulate core parameters"""
  def __init__(self, S, K, T, r, sigma, dividend=0, option_type="call", exercise_type="european"):
    """Raise ValueError if inputs are invalid"""
    clean_option_type = str(option_type).strip().lower()
    clean_exercise_type = str(exercise_type).strip().lower()
    if clean_option_type not in ["call", "put"]:
      raise ValueError(f"Invalid option_type: '{option_type}'. Expected 'call' or 'put'.")
    if clean_exercise_type not in ["european", "american"]:
      raise ValueError(f"Invalid exercise_type: '{exercise_type}'. Expected 'european' or 'american'.")

    self.S = float(S)
    self.K = float(K)
    self.T = float(T)
    self.r = float(r)
    self.sigma = float(sigma)
    self.dividend = float(dividend)
    self.option_type = clean_option_type
    self.exercise_type = clean_exercise_type

  def get_parameters(self):
    return self.S, self.K, self.T, self.r, self.sigma, self.dividend, self.option_type, self.exercise_type


class PricingEngine(ABC):
  """Abstract base class for pricing engines"""
  @abstractmethod
  def calculate(self, option: Option, **kwargs) -> float:
    # Every engine should implement this method
    pass