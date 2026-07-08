# Option Pricing Engine

A comprehensive quantitative financial pricing engine implementing advanced numerical methods and 3D risk analytics for option valuation.

Repository: [yeedong101/option-pricing-engine](https://www.google.com/search?q=https://github.com/yeedong101/option-pricing-engine)

## 1. Features & Pricing Engines

The repository encapsulates four foundational valuation models:

* **Black-Scholes Analytical Model**: Closed-form exact solutions for European options.
* **Binomial Tree Model**: Supports European and American options via iterative backward induction.
* **Monte Carlo Simulation**: Employs stochastic calculus with variance reduction (Standard, Antithetic Variates, Importance Sampling).
* **Finite Difference Method (Implicit)**: Solves the Black-Scholes PDE using numerical grid discretization and dynamic boundary conditions.

### Complexity & Optimization Analysis

| Engine | Time Complexity | Space Complexity | Optimization Notes |
| --- | --- | --- | --- |
| **Black-Scholes** | O(1) | O(1) | Executed via optimized SciPy vectorized standard normal CDF formulations. |
| **Binomial Tree** | O(N^2) | O(N) | Optimized from O(N^2) to O(N) space by continually overwriting a 1D price vector during backward induction. |
| **Monte Carlo** | O(M * P) | O(P) | Space optimized to O(P) by propagating only the current price vector rather than storing the full trajectory of all paths. |
| **Finite Difference** | O(N * M) | O(M) | Optimized from O(M^3) matrix inversion to O(M) time per step via the Thomas Algorithm (Tridiagonal Matrix Solver). |

*(Note: N = time steps, M = price nodes, P = simulation paths)*

## 2. Convergence Analysis

The validation module systematically benchmarks the numerical engines against the Black-Scholes exact analytical solution.

* **Overall Convergence**: Demonstrates the asymptotic consistency of the Binomial Tree, Finite Difference, and Monte Carlo engines converging toward the exact Black-Scholes price as computational steps scale.

![Convergence Plot](https://github.com/YeeDong101/option-pricing-engine/blob/main/images/Convergence_1.png?raw=true)

* **Variance Reduction**: Validates the efficiency of variance reduction algorithms. Antithetic variates exhibit lower standard error and faster stabilization toward the benchmark compared to standard Monte Carlo.

![Convergence Plot](https://github.com/YeeDong101/option-pricing-engine/blob/main/images/Convergence_2.png?raw=true)

* **Moneyness Effects**: Contrasts pricing convergence under Deep ITM and Deep OTM scenarios. Standard MC struggles with extreme variance for OTM options, whereas Importance Sampling dynamically shifts the drift to stabilize the variance and rapidly converge.

![Convergence Plot](https://github.com/YeeDong101/option-pricing-engine/blob/main/images/Convergence_3.png?raw=true)

## 3. Greeks 3D Visualization

The FDM engine computes full-grid analytics, extracting first, second, and third-order risk sensitivities (Greeks) to map multi-dimensional risk profiles.

* **Delta Surface**: Visualizes first-order directional risk, tracking the transition of the hedge ratio from 0 to 1 across underlying prices and varying maturities.

![3D Plot](https://github.com/YeeDong101/option-pricing-engine/blob/main/images/plot_3D_1.png?raw=true)

* **Gamma Surface**: Maps second-order convexity risk. Highlights severe pin risk where Gamma exponentially spikes at-the-money as the option approaches expiration.

![3D Plot](https://github.com/YeeDong101/option-pricing-engine/blob/main/images/plot_3D_2.png?raw=true)

* **Theta Surface**: Demonstrates first-order time decay. Illustrates the accelerating loss of extrinsic value for at-the-money options nearing maturity.

![3D Plot](https://github.com/YeeDong101/option-pricing-engine/blob/main/images/plot_3D_3.png?raw=true)

* **Gamma over Delta and Time**: Visualizes the direct interaction between directional exposure and convexity, identifying regimes that demand the highest rebalancing frequency.

![3D Plot](https://github.com/YeeDong101/option-pricing-engine/blob/main/images/plot_3D_4.png?raw=true)

* **Gamma / Theta Ratio**: Quantifies the structural trade-off in option portfolios, balancing the benefits of being long convexity against the continuous cost of time decay.

![3D Plot](https://github.com/YeeDong101/option-pricing-engine/blob/main/images/plot_3D_5.png?raw=true)

* **Speed Surface**: Represents the third-order risk (dGamma/dS). Critical for dynamic delta-hedging optimization in highly volatile markets, pinpointing where Gamma changes most aggressively.

![3D Plot](https://github.com/YeeDong101/option-pricing-engine/blob/main/images/plot_3D_6.png?raw=true)

* **Vega Surface**: Captures first-order volatility sensitivity. Demonstrates that Vega peaks exactly at-the-money and scales predominantly with the time to maturity.

![3D Plot](https://github.com/YeeDong101/option-pricing-engine/blob/main/images/plot_3D_7.png?raw=true)

## 4. Getting Started

**Prerequisites**
Ensure Python 3.x is installed. Install the numerical and plotting dependencies:

```bash
pip install -r requirements.txt

```

**Execution**
Execute the primary workflow to initialize the engines, verify put-call parity, benchmark convergence, and render all 3D HTML plots:

```bash
python main.py

```

## 5. License

This project is licensed under the MIT License.