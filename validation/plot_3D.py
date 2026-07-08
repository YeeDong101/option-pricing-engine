import numpy as np
import plotly.graph_objects as go
from pricing.base import Option
from pricing.finite_difference import FiniteDifference


def generate_and_plot_3d_greeks(option: Option, M=400, N=600, d_sigma=0.001):
    """
    Extracts and derives full-grid data from the Finite Difference engine to
    plot a comprehensive set of professional financial engineering 3D Greeks surfaces.
    Covers: Delta, Gamma, Theta, Gamma/Theta Ratio, Speed, and Vega via finite differences.
    """
    # 1. Dual-Grid Computation: Invoke FDM engine twice to compute Vega via perturbation
    S_curr, K, T_total, r, sigma, dividend, option_type, exercise_type = (
        option.get_parameters()
    )

    # Create a bumped option object with perturbed volatility (σ + Δσ)
    option_bumped = Option(
        S=S_curr,
        K=K,
        T=T_total,
        r=r,
        sigma=sigma + d_sigma,
        dividend=dividend,
        option_type=option_type,
        exercise_type=exercise_type,
    )

    fd_engine = FiniteDifference()
    grid_base = fd_engine.calculate(option, M=M, N=N, return_grid=True)
    grid_bumped = fd_engine.calculate(option_bumped, M=M, N=N, return_grid=True)

    # 2. Extract and Compute Base Greeks Grids
    S_space = grid_base["S_space"]  # Full underlying price axis (M+1,)
    time_space = np.linspace(0, T_total, N + 1)  # Full time axis (N+1,)

    delta_grid = grid_base["delta_grid"]  # (N+1, M+1)
    gamma_grid = grid_base["gamma_grid"]  # (N+1, M+1)
    theta_grid = grid_base["theta_grid"]  # (N+1, M+1)

    # [Third-Order Greeks: Speed Grid Computation (dGamma / dS)]
    dS = S_space[1] - S_space[0]
    speed_grid = np.zeros_like(gamma_grid)
    speed_grid[:, 1:M] = (
        gamma_grid[:, 2 : M + 1] - gamma_grid[:, 0 : M - 1]
    ) / (2 * dS)

    # [Perturbation Greeks: Vega Grid Computation (dV / dσ) scaled to 1% vol change]
    V_grid_base = grid_base["V_grid"]
    V_grid_bumped = grid_bumped["V_grid"]
    vega_grid = ((V_grid_bumped - V_grid_base) / d_sigma) * 0.01

    # 3. Financial Intuition Optimization: Crop grid to focus on core trading zone and de-noise
    # Price axis: Lock into 0.5 * K ~ 1.5 * K
    s_idx = np.where((S_space >= 0.5 * K) & (S_space <= 1.5 * K))[0]
    S_plot = S_space[s_idx]

    # Time axis: Skip the exact expiration day to prevent edge noise or holes caused by Theta discontinuity
    t_idx = np.where(time_space <= 0.98 * T_total)[0]
    T_plot = time_space[t_idx]

    # Slice the 2D matrices required for plotting
    delta_plot = delta_grid[t_idx[:, None], s_idx]
    gamma_plot = gamma_grid[t_idx[:, None], s_idx]
    theta_plot = theta_grid[t_idx[:, None], s_idx]
    speed_plot = speed_grid[t_idx[:, None], s_idx]
    vega_plot = vega_grid[t_idx[:, None], s_idx]

    # Create meshgrid for complex coordinate systems (e.g., Figure 4)
    S_mesh, T_mesh = np.meshgrid(S_plot, T_plot)

    print("====== Starting 3D Sensitivity Surface Plotting ======")

    # Figure 1: Delta Surface (Price S vs. Time T)
    fig1 = go.Figure(
        data=[go.Surface(x=S_plot, y=T_plot, z=delta_plot, colorscale="Viridis")]
    )
    fig1.update_layout(
        title="1. Option Delta Surface",
        scene=dict(
            xaxis_title="Underlying Price (S)",
            yaxis_title="Time to Maturity (t)",
            zaxis_title="Delta",
        ),
        template="plotly_white",
        width=800,
        height=600,
    )
    fig1.show()

    # Figure 2: Gamma Surface (Price S vs. Time T)
    fig2 = go.Figure(
        data=[go.Surface(x=S_plot, y=T_plot, z=gamma_plot, colorscale="Cividis")]
    )
    fig2.update_layout(
        title="2. Option Gamma Surface",
        scene=dict(
            xaxis_title="Underlying Price (S)",
            yaxis_title="Time to Maturity (t)",
            zaxis_title="Gamma",
        ),
        template="plotly_white",
        width=800,
        height=600,
    )
    fig2.show()

    # Figure 3: Theta Surface (Price S vs. Time T)
    fig3 = go.Figure(
        data=[go.Surface(x=S_plot, y=T_plot, z=theta_plot, colorscale="Plasma")]
    )
    fig3.update_layout(
        title="3. Option Theta Surface (Per Year)",
        scene=dict(
            xaxis_title="Underlying Price (S)",
            yaxis_title="Time to Maturity (t)",
            zaxis_title="Theta",
        ),
        template="plotly_white",
        width=800,
        height=600,
    )
    fig3.show()

    # Figure 4: Gamma Surface over Delta and Time
    fig4 = go.Figure(
        data=[
            go.Surface(
                x=delta_plot, y=T_mesh, z=gamma_plot, colorscale="Thermal"
            )
        ]
    )
    fig4.update_layout(
        title="4. Gamma Surface over Delta and Time",
        scene=dict(
            xaxis_title="Delta (Δ)",
            yaxis_title="Time to Maturity (t)",
            zaxis_title="Gamma (Γ)",
        ),
        template="plotly_white",
        width=800,
        height=600,
    )
    fig4.show()

    # Figure 5: Gamma / Theta Efficiency Ratio Surface
    epsilon = 1e-7
    gamma_over_theta = gamma_plot / (theta_plot + np.sign(theta_plot) * epsilon)
    gamma_over_theta = np.clip(gamma_over_theta, -5.0, 5.0)

    fig5 = go.Figure(
        data=[
            go.Surface(
                x=S_plot, y=T_plot, z=gamma_over_theta, colorscale="Deep"
            )
        ]
    )
    fig5.update_layout(
        title="5. Gamma / Theta Ratio Surface (Clipped)",
        scene=dict(
            xaxis_title="Underlying Price (S)",
            yaxis_title="Time to Maturity (t)",
            zaxis_title="Gamma / Theta",
        ),
        template="plotly_white",
        width=800,
        height=600,
    )
    fig5.show()
   
    # Figure 6: Speed Surface (Price S vs. Time T)
    fig6 = go.Figure(
        data=[go.Surface(x=S_plot, y=T_plot, z=speed_plot, colorscale="Balance")]
    )
    fig6.update_layout(
        title="6. Option Speed Surface (dGamma / dS)",
        scene=dict(
            xaxis_title="Underlying Price (S)",
            yaxis_title="Time to Maturity (t)",
            zaxis_title="Speed",
        ),
        template="plotly_white",
        width=800,
        height=600,
    )
    fig6.show()
    
    # Figure 7: Vega Surface (Perturbation Method - Price S vs. Time T)
    fig7 = go.Figure(
        data=[go.Surface(x=S_plot, y=T_plot, z=vega_plot, colorscale="Algae")]
    )
    fig7.update_layout(
        title="7. Option Vega Surface (Per 1% Volatility Change)",
        scene=dict(
            xaxis_title="Underlying Price (S)",
            yaxis_title="Time to Maturity (t)",
            zaxis_title="Vega (𝜈)",
        ),
        template="plotly_white",
        width=800,
        height=600,
    )
    fig7.show()


if __name__ == "__main__":
    # Create a standard European Call Option as the baseline object
    target_option = Option(
        S=100,  
        K=100,  
        T=0.5, 
        r=0.05, 
        sigma=0.25, 
        dividend=0.02, 
        option_type="call",
        exercise_type="european",
    )

    # Execute the comprehensive plotting module
    # Higher M and N values optimize surface smoothness after finite difference subtractions
    generate_and_plot_3d_greeks(target_option, M=300, N=500, d_sigma=0.001)
    
    print("Done!")