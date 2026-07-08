import numpy as np
from pricing.base import Option


class ParityChecker:
    """
    To check whether the Put-Call Parity holds for European options.
    Formula: Call - Put = S * exp(-q * T) - K * exp(-r * T)
    """

    def __init__(self, engine):
        """Initialize with the pricing engine instance to be verified."""
        self.engine = engine

    def _get_price(self, option: Option, **kwargs) -> float:
        res = self.engine.calculate(option, **kwargs)
        if isinstance(res, dict):
            return float(res["V"])
        return float(res)

    def verify(self, S, K, T, r, sigma, dividend=0, tol=1e-2, **kwargs) -> dict:
        """Execute the verification check, and 'tol' is for the tolerance for error."""
        # 1. Create European Call and Put objects with identical parameters
        call_option = Option(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            dividend=dividend,
            option_type="call",
            exercise_type="european",
        )
        put_option = Option(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            dividend=dividend,
            option_type="put",
            exercise_type="european",
        )

        # 2. Calculate Call & Put prices using the engine
        call_price = self._get_price(call_option, **kwargs)
        put_price = self._get_price(put_option, **kwargs)

        # 3. Calculate Left-Hand Side (LHS) and Right-Hand Side (RHS) of the equation
        lhs = call_price - put_price
        rhs = S * np.exp(-dividend * T) - K * np.exp(-r * T)

        # 4. Verify if the absolute error is within the tolerance
        absolute_error = abs(lhs - rhs)
        is_valid = absolute_error <= tol

        return {
            "engine_name": self.engine.__class__.__name__,
            "call_price": call_price,
            "put_price": put_price,
            "lhs_value": lhs,
            "rhs_value": rhs,
            "absolute_error": absolute_error,
            "is_valid": is_valid,
        }


if __name__ == "__main__":
    from pricing.black_scholes import BlackScholes
    from pricing.binomial_tree import BinomialTree
    from pricing.monte_carlo import MonteCarlo
    from pricing.finite_difference import FiniteDifference

    # Test parameters
    test_params = {
        "S": 100,
        "K": 105,
        "T": 0.25,
        "r": 0.04,
        "sigma": 0.3,
        "dividend": 0.02,
    }

    print("====== Starting Put-Call Parity Verification ======")

    # Test each engine
    engines = [
        (BlackScholes(), {}),
        (BinomialTree(), {"N": 5000}),
        (
            MonteCarlo(),
            {"N": 20, "num_paths": 1000000, "method": "antithetic"},
        ),
        (FiniteDifference(), {"M": 500, "N": 1000}),
    ]

    for engine, kwargs in engines:
        checker = ParityChecker(engine)
        
        # Use a wider tolerance for Monte Carlo due to random sampling errors
        tol = 5e-2 if isinstance(engine, MonteCarlo) else 1e-3

        res = checker.verify(**test_params, tol=tol, **kwargs)

        print(f"\n[Engine]: {res['engine_name']}")
        print(
            f"  -> Call Price: {res['call_price']:.4f} | Put Price: {res['put_price']:.4f}"
        )
        print(f"  -> LHS (C - P): {res['lhs_value']:.4f}")
        print(f"  -> RHS (S*e^-qT - K*e^-rT): {res['rhs_value']:.4f}")
        print(f"  -> Absolute Error: {res['absolute_error']:.6f}")
        print(
            f"  -> Result: {'[PASS]' if res['is_valid'] else '[FAIL]'}"
        )