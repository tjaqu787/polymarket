"""
Gamma CDF Fitter for Term Structure Analysis

Fits Gamma(α, β) distribution to empirical CDF via Maximum Likelihood Estimation.
Computes bootstrap credible intervals for trading signal generation.

This module extracts and simplifies the Gamma fitting logic from PoissonTimingModel,
focusing only on Gamma distributions with clear parameterization.
"""

import numpy as np
from scipy.stats import gamma
from typing import Dict, Tuple


class GammaCDFFitter:
    """
    Fit Gamma(α, β) distribution to term structure prices via MLE.

    The Gamma distribution has two parameters:
    - α (shape): Controls where probability mass concentrates
    - β (rate): Controls how fast the clock runs (β = 1/scale in scipy)

    The CDF of Gamma(α, β) is F(t; α, β) = P(X ≤ t) where X ~ Gamma(α, β)

    This class fits these parameters to market-implied CDFs and computes
    bootstrap credible intervals for uncertainty quantification.
    """

    def __init__(self):
        """Initialize the Gamma CDF fitter."""
        pass

    def cdf_to_pdf(self,
                   times: np.ndarray,
                   cdf_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert CDF to PDF using finite differences.

        Uses the approximation: f(t) ≈ [F(t+Δt) - F(t)] / Δt

        Args:
            times: Time points in years (must be sorted, strictly increasing)
            cdf_values: CDF values at each time point (must be in [0, 1] and monotonic)

        Returns:
            Tuple of (time_midpoints, pdf_values):
                - time_midpoints: Midpoints between consecutive time points
                - pdf_values: Normalized PDF values at midpoints

        Note:
            The output arrays have length len(times) - 1 since we use differences.
        """
        # Use finite differences: f(t) ≈ [F(t+Δt) - F(t)] / Δt
        pdf_values = np.diff(cdf_values) / np.diff(times)
        time_midpoints = (times[:-1] + times[1:]) / 2

        # Ensure PDF is non-negative and normalized
        pdf_values = np.maximum(pdf_values, 0)

        # Normalize to integrate to 1
        # Integral ≈ sum(pdf_values * Δt) where Δt = mean spacing
        pdf_values = pdf_values / (np.sum(pdf_values) * np.mean(np.diff(times)))

        return time_midpoints, pdf_values

    def fit(self,
            times: np.ndarray,
            cdf_values: np.ndarray) -> Dict:
        """
        Fit Gamma distribution parameters using Maximum Likelihood Estimation.

        Implementation approach:
        1. Convert CDF to PDF via finite differences
        2. Importance sample from the implied PDF (10,000 synthetic samples)
        3. Fit Gamma distribution to samples using scipy.stats.gamma.fit()
        4. Calculate goodness-of-fit metrics (RMSE, AIC)
        5. Return fitted parameters in (α, β) parameterization

        Args:
            times: Time points in years (e.g., [0.25, 0.5, 1.0, 2.0])
            cdf_values: CDF values at each time (Yes prices, in [0, 1])

        Returns:
            Dictionary with:
                - 'alpha': Shape parameter (α)
                - 'beta': Rate parameter (β = 1/scale)
                - 'rmse': Root mean squared error of CDF fit
                - 'aic': Akaike Information Criterion
                - 'log_likelihood': Log likelihood of fit
                - 'sample_times': Synthetic samples used for fitting

        Raises:
            ValueError: If fit fails or inputs are invalid
        """
        # Validate inputs
        if len(times) < 2:
            raise ValueError("Need at least 2 time points to fit")
        if not np.all(np.diff(times) > 0):
            raise ValueError("Times must be strictly increasing")
        if not np.all((cdf_values >= 0) & (cdf_values <= 1)):
            raise ValueError("CDF values must be in [0, 1]")

        # Convert CDF to PDF
        time_midpoints, pdf_values = self.cdf_to_pdf(times, cdf_values)

        # Sample from the implied distribution (importance sampling)
        # Generate synthetic data points weighted by PDF
        n_samples = 10000
        try:
            sample_times = np.random.choice(
                time_midpoints,
                size=n_samples,
                replace=True,
                p=pdf_values / pdf_values.sum()
            )
        except Exception as e:
            raise ValueError(f"Failed to sample from PDF: {e}")

        # Fit Gamma using scipy
        # scipy parameterization: shape, loc, scale
        # We force loc=0 (no shift), so Gamma(shape, scale)
        try:
            shape, loc, scale = gamma.fit(sample_times, floc=0)
        except Exception as e:
            raise ValueError(f"Gamma fit failed: {e}")

        # Convert to (α, β) parameterization
        # α = shape (same)
        # β = 1/scale (rate = inverse scale)
        alpha = shape
        beta = 1 / scale

        # Calculate log-likelihood
        log_likelihood = np.sum(gamma.logpdf(sample_times, shape, 0, scale))

        # Calculate AIC: AIC = 2k - 2*log(L)
        # k = 2 parameters (shape, scale)
        n_params = 2
        aic = 2 * n_params - 2 * log_likelihood

        # Goodness of fit: compare empirical CDF to fitted CDF
        fitted_cdf = self.predict_cdf(times, alpha, beta)
        rmse = np.sqrt(np.mean((cdf_values - fitted_cdf)**2))

        return {
            'alpha': alpha,
            'beta': beta,
            'rmse': rmse,
            'aic': aic,
            'log_likelihood': log_likelihood,
            'sample_times': sample_times,
            'shape': shape,  # Keep scipy params for reference
            'scale': scale
        }

    def predict_cdf(self,
                    times: np.ndarray,
                    alpha: float,
                    beta: float) -> np.ndarray:
        """
        Predict CDF values at given times using fitted Gamma parameters.

        Args:
            times: Time points to evaluate CDF
            alpha: Shape parameter (α)
            beta: Rate parameter (β = 1/scale)

        Returns:
            CDF values F(t; α, β) at each time point

        Note:
            Uses scipy.stats.gamma.cdf with conversion β → scale = 1/β
        """
        # Convert (α, β) to scipy parameterization
        shape = alpha
        scale = 1 / beta

        # Calculate CDF
        return gamma.cdf(times, shape, 0, scale)

    def bootstrap_credible_interval(self,
                                    times: np.ndarray,
                                    cdf_values: np.ndarray,
                                    eval_times: np.ndarray,
                                    ci_level: float = 0.70,
                                    n_bootstrap: int = 500) -> Dict:
        """
        Calculate bootstrap credible intervals for CDF predictions.

        Parametric bootstrap procedure:
        1. Convert CDF to PDF
        2. For each bootstrap iteration:
            a. Sample with replacement from implied PDF
            b. Fit Gamma to bootstrap sample
            c. Predict CDF at eval_times
        3. Compute percentiles across bootstrap samples

        Args:
            times: Original time points (training data)
            cdf_values: Original CDF values (training data)
            eval_times: Time points to evaluate CI (typically same as times)
            ci_level: Credible interval level (0.70 = 70% CI)
            n_bootstrap: Number of bootstrap samples (default 500)

        Returns:
            Dictionary with:
                - 'lower': Lower CI bound at each eval_time
                - 'upper': Upper CI bound at each eval_time
                - 'median': Median CDF at each eval_time
                - 'times': eval_times (for reference)

        Note:
            This is parametric bootstrap (refit parameters each iteration),
            NOT Bayesian posterior sampling. Captures MLE uncertainty.
        """
        # Convert CDF to PDF
        time_midpoints, pdf_values = self.cdf_to_pdf(times, cdf_values)

        bootstrap_cdfs = []
        n_samples = 10000  # Samples per bootstrap iteration

        for i in range(n_bootstrap):
            try:
                # Sample with replacement from the implied distribution
                sample_times = np.random.choice(
                    time_midpoints,
                    size=n_samples,
                    replace=True,
                    p=pdf_values / pdf_values.sum()
                )

                # Fit Gamma to bootstrap sample
                shape, loc, scale = gamma.fit(sample_times, floc=0)

                # Convert to (α, β)
                alpha = shape
                beta = 1 / scale

                # Predict CDF at eval_times
                boot_cdf = self.predict_cdf(eval_times, alpha, beta)

                bootstrap_cdfs.append(boot_cdf)
            except:
                # Skip failed fits (numerical issues can occur)
                continue

        # Check if we have enough successful fits
        if len(bootstrap_cdfs) < n_bootstrap / 2:
            # Fallback: return point estimates with no interval
            # This indicates high uncertainty or poor fit
            fit_result = self.fit(times, cdf_values)
            point_cdf = self.predict_cdf(eval_times, fit_result['alpha'], fit_result['beta'])
            return {
                'lower': point_cdf,
                'upper': point_cdf,
                'median': point_cdf,
                'times': eval_times
            }

        bootstrap_cdfs = np.array(bootstrap_cdfs)

        # Calculate percentiles
        # For ci_level = 0.70, we want 15th and 85th percentiles
        alpha = (1 - ci_level) / 2
        lower_percentile = alpha * 100
        upper_percentile = (1 - alpha) * 100

        return {
            'lower': np.percentile(bootstrap_cdfs, lower_percentile, axis=0),
            'upper': np.percentile(bootstrap_cdfs, upper_percentile, axis=0),
            'median': np.percentile(bootstrap_cdfs, 50, axis=0),
            'times': eval_times
        }
