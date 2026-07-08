import numpy as np

# Import core business logic objects (Modify paths based on your actual structure)
from pricing.base import Option
from pricing.binomial_tree import BinomialTree
from pricing.black_scholes import BlackScholes
from pricing.finite_difference import FiniteDifference
from pricing.monte_carlo import MonteCarlo

# Import your three validation modules
from validation.parity_check import ParityChecker
from validation.convergence import ConvergenceTester
from validation.plot_3D import generate_and_plot_3d_greeks


def main():
    print("====================================================")
    # 1. Setup Baseline Environment & Test Parameters
    print("[1/4] Initializing Baseline Option Parameters...")
    params = {
        "S": 100.0,
        "K": 100.0,
        "T": 0.5,
        "r": 0.05,
        "sigma": 0.25,
        "dividend": 0.02,
        "option_type": "call",
    }

    # Standard Option object used across modules
    target_option = Option(**params)

    # Initialize all 4 major pricing engines
    bs = BlackScholes()
    tree = BinomialTree()
    mc = MonteCarlo()
    fd = FiniteDifference()

    # 2. Execute Put-Call Parity Verification (Module 1)
    print("\n[2/4] Running Put-Call Parity Verification...")
    checker = ParityChecker(engine=bs)
    # Verify using Black-Scholes engine as an example
    res = checker.verify(**params, tol=1e-3)
    print(f"  -> Engine: {res['engine_name']}")
    print(f"  -> Absolute Error: {res['absolute_error']:.6f}")
    print(f"  -> Parity Check: {'[PASS]' if res['is_valid'] else '[FAIL]'}")

    # 3. Execute Convergence Analysis (Module 2)
    print("\n[3/4] Launching Convergence Tests (Plotly)...")
    tester = ConvergenceTester(bs, tree, mc, fd)

    # Trigger multi-engine and Monte Carlo specific charts
    tester.test_overall_convergence(**params)
    tester.test_mc_methods_convergence(**params)
    tester.test_mc_moneyness_convergence()

    # 4. Generate 3D Greeks Surfaces (Module 3)
    print("\n[4/4] Generating 3D Greeks Analytics Grids...")
    # This will render 7 interactive 3D surface charts via FDM engine
    generate_and_plot_3d_greeks(target_option, M=200, N=300, d_sigma=0.001)

    print("\n================ Workflow Completed ================")


if __name__ == "__main__":
    main()