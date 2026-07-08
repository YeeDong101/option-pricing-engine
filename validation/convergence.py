import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pricing.base import Option


class ConvergenceTester:
    """
    To test the convergence behavior of various pricing engines
    and compare them visually using Plotly.
    """

    def __init__(self, bs_engine, tree_engine, mc_engine, fd_engine):
        """
        Initialize with instances of the four major pricing engines.
        """
        self.bs_engine = bs_engine
        self.tree_engine = tree_engine
        self.mc_engine = mc_engine
        self.fd_engine = fd_engine

    def _get_price(self, engine, option: Option, **kwargs) -> float:
        """Standardizes differences between dictionary outputs (MC, FDM) and float outputs (BS, Tree)."""
        res = engine.calculate(option, **kwargs)
        if isinstance(res, dict):
            return float(res["V"])
        return float(res)

    def test_overall_convergence(
        self, S, K, T, r, sigma, dividend=0, option_type="call"
    ):
        """
        1. Compare the convergence profiles of different pricing methods
           using the Black-Scholes exact analytical solution as the baseline.
        """
        opt = Option(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            dividend=dividend,
            option_type=option_type,
            exercise_type="european",
        )

        # Compute exact BS analytical solution as baseline
        bs_price = self._get_price(self.bs_engine, opt)

        # Define computational complexity axis (X-axis: steps / grids / base paths)
        steps = [10, 50, 100, 200, 500, 1000, 2000, 5000]

        tree_prices = []
        fdm_prices = []
        mc_prices = []

        for n in steps:
            # Binomial Tree: converges as steps N increase
            tree_prices.append(self._get_price(self.tree_engine, opt, N=n))

            # FDM: converges as time steps N increase (space steps M fixed at 200)
            fdm_prices.append(
                self._get_price(self.fd_engine, opt, M=200, N=n)
            )

            # Monte Carlo: paths scaled to n * 100 to align with the scale of the X-axis
            mc_prices.append(
                self._get_price(
                    self.mc_engine,
                    opt,
                    N=20,
                    num_paths=n * 100,
                    method="standard",
                )
            )

        # --- Plotly Visualization ---
        fig = go.Figure()

        # Baseline
        fig.add_trace(
            go.Scatter(
                x=[steps[0], steps[-1]],
                y=[bs_price, bs_price],
                mode="lines",
                name="Black-Scholes (Exact)",
                line=dict(color="black", dash="dash", width=2),
            )
        )
        # Binomial Tree
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=tree_prices,
                mode="lines+markers",
                name="Binomial Tree (vs Steps N)",
            )
        )
        # Finite Difference
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=fdm_prices,
                mode="lines+markers",
                name="Finite Difference (vs Time N)",
            )
        )
        # Monte Carlo
        fig.add_trace(
            go.Scatter(
                x=steps,
                y=mc_prices,
                mode="lines+markers",
                name="Monte Carlo (vs Paths N*100)",
            )
        )

        fig.update_layout(
            title=f"Overall Convergence Comparison ({option_type.capitalize()} Option)",
            xaxis_title="Computation Scale / Steps",
            yaxis_title="Option Price",
            template="plotly_white",
        )
        fig.show()

    def test_mc_methods_convergence(
        self, S, K, T, r, sigma, dividend=0, option_type="call"
    ):
        """
        2. Compare convergence behavior among different Monte Carlo techniques
           (Standard vs. Antithetic Variates).
        """
        opt = Option(
            S=S,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            dividend=dividend,
            option_type=option_type,
            exercise_type="european",
        )
        bs_price = self._get_price(self.bs_engine, opt)

        # Define path sizes to test
        path_sizes = [1000, 5000, 10000, 50000, 100000, 200000, 500000]

        std_prices = []
        anti_prices = []

        for paths in path_sizes:
            std_prices.append(
                self._get_price(
                    self.mc_engine, opt, N=10, num_paths=paths, method="standard"
                )
            )
            anti_prices.append(
                self._get_price(
                    self.mc_engine,
                    opt,
                    N=10,
                    num_paths=paths,
                    method="antithetic",
                )
            )

        # Plotly Visualization
        fig = go.Figure()

        # Baseline
        fig.add_trace(
            go.Scatter(
                x=[path_sizes[0], path_sizes[-1]],
                y=[bs_price, bs_price],
                mode="lines",
                name="Black-Scholes Benchmark",
                line=dict(color="black", dash="dash"),
            )
        )
        # Standard MC
        fig.add_trace(
            go.Scatter(
                x=path_sizes,
                y=std_prices,
                mode="lines+markers",
                name="Standard MC",
                line=dict(color="red"),
            )
        )
        # Antithetic MC
        fig.add_trace(
            go.Scatter(
                x=path_sizes,
                y=anti_prices,
                mode="lines+markers",
                name="Antithetic MC",
                line=dict(color="blue"),
            )
        )

        fig.update_layout(
            title="Monte Carlo Convergence: Standard vs Antithetic Variates",
            xaxis_title="Number of Paths",
            yaxis_title="Option Price",
            xaxis_type="log",  # Log scale highlights diminishing variance at high path counts
            template="plotly_white",
        )
        fig.show()

    def test_mc_moneyness_convergence(
        self, K=100, T=1.0, r=0.05, sigma=0.2, dividend=0.01
    ):
        """
        3. Compare convergence profiles under Deep ITM / Deep OTM scenarios.
           Evaluates: Standard vs. Antithetic vs. Importance Sampling.
        """
        # Define spot prices for deep in-the-money and deep out-of-the-money
        S_itm = 140.0
        S_otm = 60.0

        opt_itm = Option(
            S=S_itm,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            dividend=dividend,
            option_type="call",
        )
        opt_otm = Option(
            S=S_otm,
            K=K,
            T=T,
            r=r,
            sigma=sigma,
            dividend=dividend,
            option_type="call",
        )

        # Calculate baselines using self.bs_engine
        bs_price_itm = self._get_price(self.bs_engine, opt_itm)
        bs_price_otm = self._get_price(self.bs_engine, opt_otm)

        path_sizes = [1000, 5000, 10000, 30000, 50000, 100000, 200000]
        methods = ["standard", "antithetic", "importance"]

        results = {
            "itm": {m: [] for m in methods},
            "otm": {m: [] for m in methods},
        }

        # Run pricing simulations using self.mc_engine
        for paths in path_sizes:
            for m in methods:
                results["itm"][m].append(
                    self._get_price(
                        self.mc_engine,
                        opt_itm,
                        N=10,
                        num_paths=paths,
                        method=m,
                    )
                )
                results["otm"][m].append(
                    self._get_price(
                        self.mc_engine,
                        opt_otm,
                        N=10,
                        num_paths=paths,
                        method=m,
                    )
                )

        # Create interactive 1x2 subplots
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=(
                f"Deep ITM Call (S={S_itm}, K={K})",
                f"Deep OTM Call (S={S_otm}, K={K})",
            ),
        )

        colors = {
            "standard": "#EF553B",
            "antithetic": "#636efa",
            "importance": "#00cc96",
        }

        # Left Plot: Deep ITM
        fig.add_trace(
            go.Scatter(
                x=[path_sizes[0], path_sizes[-1]],
                y=[bs_price_itm, bs_price_itm],
                mode="lines",
                name="BS Benchmark (ITM)",
                line=dict(color="black", dash="dash"),
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        for m in methods:
            fig.add_trace(
                go.Scatter(
                    x=path_sizes,
                    y=results["itm"][m],
                    mode="lines+markers",
                    name=f"{m.capitalize()} MC",
                    line=dict(color=colors[m]),
                    showlegend=False,
                ),
                row=1,
                col=1,
            )

        # Right Plot: Deep OTM
        fig.add_trace(
            go.Scatter(
                x=[path_sizes[0], path_sizes[-1]],
                y=[bs_price_otm, bs_price_otm],
                mode="lines",
                name="BS Benchmark",
                line=dict(color="black", dash="dash"),
            ),
            row=1,
            col=2,
        )
        for m in methods:
            fig.add_trace(
                go.Scatter(
                    x=path_sizes,
                    y=results["otm"][m],
                    mode="lines+markers",
                    name=f"{m.capitalize()} MC",
                    line=dict(color=colors[m]),
                ),
                row=1,
                col=2,
            )

        # Format layout and axes
        fig.update_xaxes(
            type="log", title_text="Number of Paths (Log Scale)", row=1, col=1
        )
        fig.update_xaxes(
            type="log", title_text="Number of Paths (Log Scale)", row=1, col=2
        )
        fig.update_yaxes(title_text="Option Price", row=1, col=1)
        fig.update_yaxes(title_text="Option Price", row=1, col=2)

        fig.update_layout(
            title="<b>Monte Carlo Convergence Analysis: Moneyness Effects</b>",
            template="plotly_white",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=1.02),
            width=1100,
            height=500,
        )
        fig.show()


if __name__ == "__main__":
    from pricing.black_scholes import BlackScholes
    from pricing.binomial_tree import BinomialTree
    from pricing.monte_carlo import MonteCarlo
    from pricing.finite_difference import FiniteDifference

    # Initialize pricing engines
    bs = BlackScholes()
    tree = BinomialTree()
    mc = MonteCarlo()
    fd = FiniteDifference()

    # Instantiate the convergence tester
    tester = ConvergenceTester(bs, tree, mc, fd)

    # Base test parameters
    params = {
        "S": 100,
        "K": 105,
        "T": 1.0,
        "r": 0.05,
        "sigma": 0.2,
        "dividend": 0.01,
        "option_type": "call",
    }

    print("Running: Overall convergence comparison...")
    tester.test_overall_convergence(**params)

    print("Running: Monte Carlo method variations convergence...")
    tester.test_mc_methods_convergence(**params)

    print("Running: Moneyness effects on Monte Carlo convergence...")
    tester.test_mc_moneyness_convergence()
    
    print("Done!")