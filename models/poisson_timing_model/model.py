"""
Poisson Timing Model for Polymarket Event Timing Prediction

SICK IDEA: Treat the term structure of prices as a CDF!

For time-distributed contracts like:
- "By March 2026" (price = 0.2)
- "By June 2026" (price = 0.5)
- "By December 2026" (price = 0.9)

These prices form a CDF: F(t) = P(event happens by time t)

Then:
1. Take derivative to get PDF: f(t) = dF/dt
2. Fit a Poisson process: events arrive with rate λ
3. For Poisson, inter-arrival times ~ Exponential(λ)
4. Or use Weibull/Gamma for more flexibility
5. Estimate λ from the market's implied distribution
6. Predict the most likely time bucket

This is way cleaner than hierarchical priors - just inverse transform sampling!
"""

import pymc as pm
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
import arviz as az
from scipy.interpolate import interp1d
from scipy.stats import expon, poisson, gamma, weibull_min
from scipy.optimize import minimize
from scipy.special import gamma as gamma_func


class PoissonTimingModel:
    """
    Fit a Poisson process to market-implied timing distributions.

    Given prices for "by time t" contracts, treat as CDF and estimate
    the rate parameter λ of the underlying Poisson process.
    """

    def __init__(self, distribution: str = 'exponential'):
        """
        Initialize the model.

        Args:
            distribution: Type of distribution ('exponential', 'weibull', 'gamma')
                - exponential: Poisson process with constant rate λ
                - weibull: More flexible, allows for time-varying hazard
                - gamma: Continuous analog of negative binomial
        """
        self.distribution = distribution
        self.model = None
        self.trace = None

    def extract_cdf_from_prices(self,
                                prices_df: pd.DataFrame,
                                time_col: str = 'time_to_expiration',
                                price_col: str = 'price') -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract CDF from market prices.

        Args:
            prices_df: DataFrame with time and price columns for one event
            time_col: Column name for time/maturity
            price_col: Column name for prices

        Returns:
            Tuple of (times, cdf_values) sorted by time
        """
        # Sort by time
        df_sorted = prices_df.sort_values(time_col)

        times = df_sorted[time_col].values
        cdf_values = df_sorted[price_col].values

        # Ensure CDF is monotonic and in [0, 1]
        cdf_values = np.clip(cdf_values, 0, 1)

        return times, cdf_values

    def cdf_to_pdf(self,
                   times: np.ndarray,
                   cdf_values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert CDF to PDF using finite differences.

        Args:
            times: Time points
            cdf_values: CDF values at each time point

        Returns:
            Tuple of (time_midpoints, pdf_values)
        """
        # Use finite differences: f(t) ≈ [F(t+Δt) - F(t)] / Δt
        pdf_values = np.diff(cdf_values) / np.diff(times)
        time_midpoints = (times[:-1] + times[1:]) / 2

        # Ensure PDF is non-negative and normalized
        pdf_values = np.maximum(pdf_values, 0)
        pdf_values = pdf_values / (np.sum(pdf_values) * np.mean(np.diff(times)))  # Normalize

        return time_midpoints, pdf_values

    def fit_mle(self,
                times: np.ndarray,
                cdf_values: np.ndarray) -> Dict:
        """
        Fit distribution parameters using Maximum Likelihood Estimation.

        Args:
            times: Time points
            cdf_values: CDF values (market prices)

        Returns:
            Dictionary with fitted parameters and goodness-of-fit metrics
        """
        # Convert CDF to PDF
        time_midpoints, pdf_values = self.cdf_to_pdf(times, cdf_values)

        # Sample from the implied distribution (importance sampling)
        # Generate synthetic data points weighted by PDF
        n_samples = 10000
        sample_times = np.random.choice(
            time_midpoints,
            size=n_samples,
            p=pdf_values / pdf_values.sum()
        )

        # Fit distribution to samples
        if self.distribution == 'exponential':
            # For exponential: λ = 1/mean
            lambda_hat = 1 / np.mean(sample_times)
            params = {'lambda': lambda_hat}

            # Calculate log-likelihood
            log_likelihood = np.sum(expon.logpdf(sample_times, scale=1/lambda_hat))

        elif self.distribution == 'weibull':
            # Fit Weibull using scipy
            shape, loc, scale = weibull_min.fit(sample_times, floc=0)
            params = {'shape': shape, 'scale': scale}

            log_likelihood = np.sum(weibull_min.logpdf(sample_times, shape, 0, scale))

        elif self.distribution == 'gamma':
            # Fit Gamma using scipy
            shape, loc, scale = gamma.fit(sample_times, floc=0)
            params = {'shape': shape, 'scale': scale}

            log_likelihood = np.sum(gamma.logpdf(sample_times, shape, 0, scale))

        else:
            raise ValueError(f"Unknown distribution: {self.distribution}")

        # Calculate AIC and BIC
        n_params = len(params)
        aic = 2 * n_params - 2 * log_likelihood
        bic = n_params * np.log(len(sample_times)) - 2 * log_likelihood

        # Goodness of fit: compare empirical CDF to fitted CDF
        fitted_cdf = self._calculate_fitted_cdf(times, params)
        rmse = np.sqrt(np.mean((cdf_values - fitted_cdf)**2))

        return {
            'params': params,
            'log_likelihood': log_likelihood,
            'aic': aic,
            'bic': bic,
            'rmse': rmse,
            'sample_times': sample_times
        }

    def _calculate_fitted_cdf(self, times: np.ndarray, params: Dict) -> np.ndarray:
        """Calculate CDF from fitted distribution parameters."""
        if self.distribution == 'exponential':
            return 1 - np.exp(-params['lambda'] * times)
        elif self.distribution == 'weibull':
            return weibull_min.cdf(times, params['shape'], 0, params['scale'])
        elif self.distribution == 'gamma':
            return gamma.cdf(times, params['shape'], 0, params['scale'])

    def build_bayesian_model(self,
                            times: np.ndarray,
                            cdf_values: np.ndarray) -> pm.Model:
        """
        Build a Bayesian model to estimate distribution parameters.

        Args:
            times: Time points
            cdf_values: CDF values (market prices)

        Returns:
            PyMC model
        """
        with pm.Model() as model:
            if self.distribution == 'exponential':
                # Prior on rate parameter λ
                lambda_ = pm.HalfNormal('lambda', sigma=2)

                # CDF of exponential: F(t) = 1 - exp(-λt)
                mu = 1 - pm.math.exp(-lambda_ * times)

            elif self.distribution == 'weibull':
                # Priors on shape and scale
                shape = pm.HalfNormal('shape', sigma=2)
                scale = pm.HalfNormal('scale', sigma=2)

                # CDF of Weibull: F(t) = 1 - exp(-(t/scale)^shape)
                mu = 1 - pm.math.exp(-pm.math.pow(times / scale, shape))

            elif self.distribution == 'gamma':
                # Priors on shape and scale
                shape = pm.HalfNormal('shape', sigma=2)
                scale = pm.HalfNormal('scale', sigma=2)

                # For Gamma, use incomplete gamma function approximation
                # This is tricky in PyMC, so we'll use a simpler approximation
                # For now, use exponential as a special case
                lambda_ = shape / scale
                mu = 1 - pm.math.exp(-lambda_ * times)

            # Likelihood: observed CDF values with some noise
            # Model as Beta-distributed around the fitted CDF
            # Use concentration parameter to control tightness
            kappa = pm.HalfNormal('kappa', sigma=100)

            # Clip mu to (0, 1) for Beta distribution
            mu_clipped = pm.math.clip(mu, 0.001, 0.999)

            # Beta distribution parameterized by mean and concentration
            alpha = mu_clipped * kappa
            beta = (1 - mu_clipped) * kappa

            obs = pm.Beta('obs',
                         alpha=alpha,
                         beta=beta,
                         observed=cdf_values)

        self.model = model
        return model

    def fit_bayesian(self,
                    times: np.ndarray,
                    cdf_values: np.ndarray,
                    draws: int = 2000,
                    tune: int = 1000,
                    chains: int = 4) -> az.InferenceData:
        """
        Fit model using Bayesian inference (MCMC).

        Args:
            times: Time points
            cdf_values: CDF values
            draws: Number of MCMC samples
            tune: Number of tuning steps
            chains: Number of chains

        Returns:
            ArviZ InferenceData object
        """
        if self.model is None:
            self.build_bayesian_model(times, cdf_values)

        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=0.95,
                return_inferencedata=True
            )

        return self.trace

    def predict_timing(self,
                      params: Dict,
                      times: np.ndarray) -> Dict:
        """
        Predict the most likely timing and distribution.

        Args:
            params: Fitted distribution parameters
            times: Time points to evaluate

        Returns:
            Dictionary with predictions
        """
        # Calculate PDF at each time point
        if self.distribution == 'exponential':
            lambda_ = params['lambda']
            pdf = lambda_ * np.exp(-lambda_ * times)
            cdf = 1 - np.exp(-lambda_ * times)
            mode = 0  # Exponential mode is always at t=0
            mean = 1 / lambda_
            median = np.log(2) / lambda_

        elif self.distribution == 'weibull':
            shape = params['shape']
            scale = params['scale']
            pdf = weibull_min.pdf(times, shape, 0, scale)
            cdf = weibull_min.cdf(times, shape, 0, scale)

            # Weibull mode
            if shape > 1:
                mode = scale * ((shape - 1) / shape) ** (1 / shape)
            else:
                mode = 0
            mean = scale * gamma_func(1 + 1/shape)
            median = scale * (np.log(2)) ** (1/shape)

        elif self.distribution == 'gamma':
            shape = params['shape']
            scale = params['scale']
            pdf = gamma.pdf(times, shape, 0, scale)
            cdf = gamma.cdf(times, shape, 0, scale)

            # Gamma mode
            if shape >= 1:
                mode = (shape - 1) * scale
            else:
                mode = 0
            mean = shape * scale
            # Gamma median doesn't have closed form
            median = gamma.ppf(0.5, shape, 0, scale)

        # Find most likely time bucket
        most_likely_idx = np.argmax(pdf)
        most_likely_time = times[most_likely_idx]

        return {
            'pdf': pdf,
            'cdf': cdf,
            'most_likely_time': most_likely_time,
            'mode': mode,
            'mean': mean,
            'median': median,
            'times': times
        }

    def compare_distributions(self,
                            times: np.ndarray,
                            cdf_values: np.ndarray) -> pd.DataFrame:
        """
        Compare multiple distribution types and return best fit.

        Args:
            times: Time points
            cdf_values: CDF values

        Returns:
            DataFrame with comparison metrics
        """
        results = []

        for dist in ['exponential', 'weibull', 'gamma']:
            model = PoissonTimingModel(distribution=dist)
            fit_result = model.fit_mle(times, cdf_values)

            results.append({
                'distribution': dist,
                'aic': fit_result['aic'],
                'bic': fit_result['bic'],
                'rmse': fit_result['rmse'],
                'log_likelihood': fit_result['log_likelihood'],
                **fit_result['params']
            })

        df = pd.DataFrame(results)
        df = df.sort_values('aic')  # Best model has lowest AIC

        return df


def demo_model():
    """Demonstrate the model with synthetic data."""

    print("="*70)
    print("POISSON TIMING MODEL - DEMO")
    print("="*70)

    # Create synthetic CDF (true λ = 0.5)
    true_lambda = 0.5
    times = np.array([0.1, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0])
    true_cdf = 1 - np.exp(-true_lambda * times)

    # Add some noise
    np.random.seed(42)
    noisy_cdf = np.clip(true_cdf + np.random.normal(0, 0.05, len(times)), 0, 1)

    print(f"\nTrue λ: {true_lambda}")
    print(f"Times: {times}")
    print(f"True CDF: {true_cdf}")
    print(f"Noisy CDF: {noisy_cdf}")

    # Compare distributions
    print("\n" + "="*70)
    print("COMPARING DISTRIBUTIONS (MLE)")
    print("="*70)

    model = PoissonTimingModel()
    comparison = model.compare_distributions(times, noisy_cdf)
    print(comparison.to_string())

    # Fit best model
    best_dist = comparison.iloc[0]['distribution']
    print(f"\n" + "="*70)
    print(f"FITTING BEST MODEL: {best_dist}")
    print("="*70)

    model = PoissonTimingModel(distribution=best_dist)
    fit_result = model.fit_mle(times, noisy_cdf)

    print(f"\nFitted parameters: {fit_result['params']}")
    print(f"RMSE: {fit_result['rmse']:.4f}")

    # Make predictions
    print("\n" + "="*70)
    print("PREDICTIONS")
    print("="*70)

    pred_times = np.linspace(0, 6, 100)
    predictions = model.predict_timing(fit_result['params'], pred_times)

    print(f"Most likely time: {predictions['most_likely_time']:.3f}")
    print(f"Mode: {predictions['mode']:.3f}")
    print(f"Mean: {predictions['mean']:.3f}")
    print(f"Median: {predictions['median']:.3f}")


if __name__ == "__main__":
    demo_model()
