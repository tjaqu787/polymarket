"""
Kelly Criterion Position Sizing for Bayesian Carry Strategy

Uses posterior uncertainty from Bayesian Gamma model to size positions
via the Kelly criterion.

Key Concept:
- Edge = P_model(No wins) - P_market(No wins)
- Kelly fraction: f* = (μ_edge / σ²_edge) * kelly_fraction
- Wide posterior (high variance) → small position (uncertain)
- Narrow posterior (low variance) → large position (confident)
"""

import numpy as np
from scipy.stats import gamma as scipy_gamma
from typing import Tuple
from .bayesian_gamma_fitter import BayesianFitResult


def get_posterior_cdf_samples(
    fit_result: BayesianFitResult,
    times: np.ndarray,
    n_samples: int = 1000
) -> np.ndarray:
    """
    Extract posterior CDF samples from Bayesian fit result.

    Args:
        fit_result: Result from BayesianGammaFitter containing idata
        times: Time points to evaluate CDF at (in years)
        n_samples: Number of posterior samples to extract

    Returns:
        Array of shape (n_samples, len(times)) containing CDF values
        for each posterior sample

    Example:
        >>> fit_result = model.fit_event(...)
        >>> times = np.array([0.1, 0.2, 0.3])  # days/365.25
        >>> cdf_samples = get_posterior_cdf_samples(fit_result, times, 1000)
        >>> cdf_samples.shape
        (1000, 3)
    """
    # Extract posterior samples from idata
    # Format: idata.posterior['alpha'] has shape (chains, draws)
    alpha_samples = fit_result.idata.posterior['alpha'].values.flatten()
    beta_samples = fit_result.idata.posterior['beta'].values.flatten()

    # Limit to n_samples
    total_samples = len(alpha_samples)
    if total_samples > n_samples:
        alpha_samples = alpha_samples[:n_samples]
        beta_samples = beta_samples[:n_samples]

    # Compute CDF for each (alpha, beta) sample
    # Parameterization: alpha = shape, beta = rate, scale = 1/beta
    cdf_samples = np.array([
        scipy_gamma.cdf(times, a=alpha, scale=1/beta)
        for alpha, beta in zip(alpha_samples, beta_samples)
    ])

    return cdf_samples


def edge_distribution(
    posterior_cdf_samples: np.ndarray,
    market_prices: np.ndarray,
    tenor_idx: int
) -> Tuple[float, float, np.ndarray]:
    """
    Calculate edge distribution for a specific tenor.

    Edge is defined as:
        edge = P_model(No wins) - P_market(No wins)

    For a timing market:
        - CDF(t) = P(event happens by time t) = P(Yes)
        - 1 - CDF(t) = P(event doesn't happen by time t) = P(No)
        - Market prices No contracts
        - Positive edge = market underpricing No = BUY opportunity

    Args:
        posterior_cdf_samples: Posterior CDF samples, shape (n_samples, n_tenors)
        market_prices: Market prices for No contracts, shape (n_tenors,)
        tenor_idx: Index of tenor to calculate edge for

    Returns:
        (mu_edge, var_edge, edge_samples)
        - mu_edge: Mean edge (expected profit per dollar)
        - var_edge: Variance of edge (uncertainty)
        - edge_samples: All edge samples for diagnostics

    Example:
        >>> # Model says P(No) = 0.60±0.10, market prices No at 0.50
        >>> # Edge = 0.60 - 0.50 = 0.10 (10% expected profit)
        >>> mu, var, samples = edge_distribution(cdf_samples, prices, 0)
        >>> mu
        0.10
    """
    # P(No) = 1 - P(Yes) = 1 - CDF
    model_no_prob_samples = 1.0 - posterior_cdf_samples[:, tenor_idx]
    market_no_prob = market_prices[tenor_idx]

    # Edge: how much we expect to profit
    # Positive = model thinks No is underpriced
    # Negative = model thinks No is overpriced
    edge_samples = model_no_prob_samples - market_no_prob

    mu_edge = np.mean(edge_samples)
    var_edge = np.var(edge_samples)

    return mu_edge, var_edge, edge_samples


def kelly_size(
    posterior_cdf_samples: np.ndarray,
    market_prices: np.ndarray,
    tenor_idx: int,
    portfolio_value: float,
    fraction: float = 0.25,
    max_position: float = 0.10,
    min_edge: float = 0.05
) -> float:
    """
    Calculate fractional Kelly position size.

    The Kelly criterion optimally sizes bets to maximize log wealth.
    For uncertain edges (Bayesian posterior), we use:

        f* = fraction * (μ_edge / σ²_edge)

    Key properties:
    - Higher mean edge → larger position
    - Higher variance (uncertainty) → smaller position
    - Fractional Kelly (default 0.25) reduces risk of overbetting

    Args:
        posterior_cdf_samples: Posterior CDF samples, shape (n_samples, n_tenors)
        market_prices: Market prices for No contracts, shape (n_tenors,)
        tenor_idx: Index of tenor to size
        portfolio_value: Current portfolio value in dollars
        fraction: Kelly fraction (0.25 = quarter Kelly, conservative)
        max_position: Maximum position as fraction of portfolio (hard cap)
        min_edge: Minimum edge to take position

    Returns:
        Position size in dollars
        - 0 if edge < min_edge or variance too small
        - Capped at max_position * portfolio_value

    Example:
        >>> # Portfolio = $10,000
        >>> # Mean edge = 0.10 (10%), variance = 0.01
        >>> # f* = 0.25 * (0.10 / 0.01) = 2.5 (250%)
        >>> # Capped at max_position = 0.10 → 10% = $1,000
        >>> size = kelly_size(cdf_samples, prices, 0, 10000)
        >>> size
        1000.0
    """
    # Calculate edge distribution
    mu_edge, var_edge, _ = edge_distribution(
        posterior_cdf_samples, market_prices, tenor_idx
    )

    # No position if:
    # 1. Edge below minimum threshold
    # 2. Variance too small (numerical instability)
    if mu_edge < min_edge or var_edge < 1e-8:
        return 0.0

    # Kelly fraction: f = (μ_edge / σ²_edge)
    # High variance → small f (uncertain, size down)
    # Low variance → large f (confident, size up)
    f = fraction * (mu_edge / var_edge)

    # Clip to valid range [0, max_position]
    f = np.clip(f, 0.0, max_position)

    # Convert fraction to dollars
    size_dollars = f * portfolio_value

    return size_dollars
