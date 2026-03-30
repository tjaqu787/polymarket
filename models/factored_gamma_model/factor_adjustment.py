"""
Factor Adjustment Module

Applies factor adjustments to base Gamma parameters in log space.
This ensures parameters remain strictly positive while allowing for
multiplicative adjustments based on category and term structure features.
"""

import numpy as np
from typing import Tuple


class FactorAdjustment:
    """
    Apply factor adjustments to base Gamma parameters.

    The adjustment formula (from instructions.txt):
        log_α = log(α_base) + θ_shape[category] + β_curv * curvature
        log_β = log(β_base) + φ_rate[category] + β_slope * slope + β_rate * implied_rate

    Where:
        - θ_shape[category]: Per-category adjustment to shape
        - φ_rate[category]: Per-category adjustment to rate
        - β_curv, β_slope, β_rate: Global feature coefficients

    Log-space adjustments ensure:
        1. Parameters remain strictly positive (α, β > 0)
        2. Adjustments are multiplicative in original space
        3. Stable numerical behavior

    Example:
        If θ_shape[politics] = 0.15, then α_adj = α_base * exp(0.15) ≈ 1.16 * α_base
        This represents a 16% increase in the shape parameter for politics events.
    """

    @staticmethod
    def adjust(
        alpha_base: float,
        beta_base: float,
        category: str,
        ts_slope: float,
        ts_curvature: float,
        implied_rate: float,
        eb_factors: 'EmpiricalBayesFactors'
    ) -> Tuple[float, float]:
        """
        Apply factor adjustments to base Gamma parameters.

        Args:
            alpha_base: Base shape parameter from MLE fit
            beta_base: Base rate parameter from MLE fit
            category: Event category (e.g., 'politics', 'crypto', 'sports')
            ts_slope: Term structure slope (steepness of price curve)
            ts_curvature: Term structure curvature (bowing of price curve)
            implied_rate: Implied discount rate from term structure
            eb_factors: Fitted EmpiricalBayesFactors instance

        Returns:
            Tuple of (alpha_adjusted, beta_adjusted)

        Raises:
            ValueError: If alpha_base or beta_base <= 0

        Note:
            If features are NaN or missing, they are treated as 0 (no adjustment).
            If category is unknown, default adjustment is applied.
        """
        # Validate inputs
        if alpha_base <= 0:
            raise ValueError(f"alpha_base must be positive, got {alpha_base}")
        if beta_base <= 0:
            raise ValueError(f"beta_base must be positive, got {beta_base}")

        # Handle NaN features (treat as 0)
        ts_slope = 0.0 if np.isnan(ts_slope) else ts_slope
        ts_curvature = 0.0 if np.isnan(ts_curvature) else ts_curvature
        implied_rate = 0.0 if np.isnan(implied_rate) else implied_rate

        # Get factor adjustments from empirical Bayes
        adjustments = eb_factors.get_factor_adjustments(
            category=category,
            ts_slope=ts_slope,
            ts_curvature=ts_curvature,
            implied_rate=implied_rate
        )

        log_alpha_adjustment = adjustments['log_alpha_adjustment']
        log_beta_adjustment = adjustments['log_beta_adjustment']

        # Apply adjustments in log space
        log_alpha_adj = np.log(alpha_base) + log_alpha_adjustment
        log_beta_adj = np.log(beta_base) + log_beta_adjustment

        # Clip log-space values to prevent overflow/underflow
        # This ensures adjusted params stay in reasonable range [1e-6, 1e6]
        log_alpha_adj = np.clip(log_alpha_adj, np.log(1e-6), np.log(1e6))
        log_beta_adj = np.clip(log_beta_adj, np.log(1e-6), np.log(1e6))

        # Exponentiate back to original space
        alpha_adj = np.exp(log_alpha_adj)
        beta_adj = np.exp(log_beta_adj)

        return alpha_adj, beta_adj

    @staticmethod
    def validate_adjusted_params(
        alpha_adj: float,
        beta_adj: float,
        min_value: float = 1e-6,
        max_value: float = 1e6
    ) -> Tuple[bool, str]:
        """
        Validate adjusted parameters are in reasonable range.

        Args:
            alpha_adj: Adjusted shape parameter
            beta_adj: Adjusted rate parameter
            min_value: Minimum allowed value
            max_value: Maximum allowed value

        Returns:
            Tuple of (is_valid, message)
        """
        if alpha_adj < min_value or alpha_adj > max_value:
            return False, f"alpha_adj={alpha_adj:.2e} outside [{min_value:.2e}, {max_value:.2e}]"

        if beta_adj < min_value or beta_adj > max_value:
            return False, f"beta_adj={beta_adj:.2e} outside [{min_value:.2e}, {max_value:.2e}]"

        return True, "Parameters valid"
