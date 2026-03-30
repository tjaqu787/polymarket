"""
Bayesian Gamma CDF Fitter using PyMC

Fits Gamma(α, β) distribution to term structure CDF using MCMC.
Supports sequential Bayesian updating: uses previous posterior as new prior.

Key Features:
- fit_initial(): First fit with weakly informative priors
- fit_sequential(): Sequential fit using previous posterior samples as prior
- predict_cdf(): Posterior predictive credible intervals

Comparison to frequentist approach:
- Frequentist (MLE + bootstrap): refits from scratch, bootstrap for CI
- Bayesian (MCMC + sequential): uses previous fit as prior, posterior for CI
"""

import numpy as np
import pymc as pm
import arviz as az
from scipy.stats import gamma as scipy_gamma
from typing import Dict, Optional, Tuple
from dataclasses import dataclass


@dataclass
class BayesianFitResult:
    """Result from Bayesian Gamma fit."""
    idata: az.InferenceData      # Posterior samples
    alpha_mean: float             # Posterior mean of α
    beta_mean: float              # Posterior mean of β
    alpha_std: float              # Posterior std of α
    beta_std: float               # Posterior std of β
    rhat_alpha: float             # Convergence diagnostic for α
    rhat_beta: float              # Convergence diagnostic for β
    ess_alpha: float              # Effective sample size for α
    ess_beta: float               # Effective sample size for β
    converged: bool               # True if R-hat < 1.01 for all params


class BayesianGammaFitter:
    """
    Fit Gamma(α, β) to term structure using Bayesian MCMC.

    Supports sequential updating: previous posterior → new prior.
    """

    def __init__(
        self,
        mcmc_draws: int = 500,
        mcmc_tune: int = 500,
        mcmc_chains: int = 2,
        mcmc_cores: int = 4,
        target_accept: float = 0.95
    ):
        """
        Initialize Bayesian Gamma fitter.

        Args:
            mcmc_draws: Number of posterior samples per chain
            mcmc_tune: Number of tuning steps
            mcmc_chains: Number of MCMC chains
            mcmc_cores: Number of CPU cores to use for parallel sampling
            target_accept: Target acceptance rate for NUTS
        """
        self.mcmc_draws = mcmc_draws
        self.mcmc_tune = mcmc_tune
        self.mcmc_chains = mcmc_chains
        self.mcmc_cores = mcmc_cores
        self.target_accept = target_accept

    def cdf_to_pdf(
        self,
        times: np.ndarray,
        cdf_values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert CDF to PDF via finite differences.

        Same as frequentist version - needed for sampling.
        """
        pdf_values = np.diff(cdf_values) / np.diff(times)
        time_midpoints = (times[:-1] + times[1:]) / 2

        pdf_values = np.maximum(pdf_values, 0)
        pdf_values = pdf_values / (np.sum(pdf_values) * np.mean(np.diff(times)))

        return time_midpoints, pdf_values

    def fit_initial(
        self,
        times: np.ndarray,
        cdf_values: np.ndarray
    ) -> Optional[BayesianFitResult]:
        """
        Initial Bayesian fit with weakly informative priors.

        Uses default priors:
        - α ~ HalfNormal(5, 3)  # Shape parameter
        - β ~ HalfNormal(1, 2)  # Rate parameter

        Args:
            times: Time points in years
            cdf_values: CDF values at each time

        Returns:
            BayesianFitResult with posterior samples, or None if fit fails
        """
        # Sample from implied PDF
        time_midpoints, pdf_values = self.cdf_to_pdf(times, cdf_values)

        n_samples = 10000
        try:
            sample_times = np.random.choice(
                time_midpoints,
                size=n_samples,
                replace=True,
                p=pdf_values / pdf_values.sum()
            )
        except Exception as e:
            print(f"Failed to sample from PDF: {e}")
            return None

        # Build PyMC model
        with pm.Model() as model:
            # Weakly informative priors
            alpha = pm.HalfNormal('alpha', sigma=5.0)
            beta = pm.HalfNormal('beta', sigma=2.0)

            # Likelihood: observed samples from Gamma distribution
            pm.Gamma('obs', alpha=alpha, beta=beta, observed=sample_times)

            # Sample posterior
            try:
                idata = pm.sample(
                    draws=self.mcmc_draws,
                    tune=self.mcmc_tune,
                    chains=self.mcmc_chains,
                    cores=self.mcmc_cores,  # Use multiple cores for parallel sampling
                    target_accept=self.target_accept,
                    progressbar=False,
                    return_inferencedata=True
                )
            except Exception as e:
                print(f"MCMC sampling failed: {e}")
                return None

        return self._process_inference_data(idata)

    def fit_sequential(
        self,
        times: np.ndarray,
        cdf_values: np.ndarray,
        prior_idata: az.InferenceData
    ) -> Optional[BayesianFitResult]:
        """
        Sequential Bayesian fit using previous posterior as prior.

        Constructs informative prior from previous posterior samples:
        - α ~ LogNormal(μ_α, σ_α) where μ_α, σ_α from previous α samples
        - β ~ LogNormal(μ_β, σ_β) where μ_β, σ_β from previous β samples

        Args:
            times: Time points in years
            cdf_values: CDF values at each time
            prior_idata: InferenceData from previous fit

        Returns:
            BayesianFitResult with updated posterior, or None if fit fails
        """
        # Extract previous posterior samples
        prev_alpha = prior_idata.posterior['alpha'].values.flatten()
        prev_beta = prior_idata.posterior['beta'].values.flatten()

        # Fit log-space parameters for LogNormal prior
        log_alpha = np.log(prev_alpha)
        log_beta = np.log(prev_beta)

        alpha_prior_mu = np.mean(log_alpha)
        alpha_prior_sigma = np.std(log_alpha)

        beta_prior_mu = np.mean(log_beta)
        beta_prior_sigma = np.std(log_beta)

        # Prevent zero variance (use weak prior if posterior was too confident)
        alpha_prior_sigma = max(alpha_prior_sigma, 0.1)
        beta_prior_sigma = max(beta_prior_sigma, 0.1)

        # Sample from implied PDF
        time_midpoints, pdf_values = self.cdf_to_pdf(times, cdf_values)

        n_samples = 10000
        try:
            sample_times = np.random.choice(
                time_midpoints,
                size=n_samples,
                replace=True,
                p=pdf_values / pdf_values.sum()
            )
        except Exception as e:
            print(f"Failed to sample from PDF: {e}")
            return None

        # Build PyMC model with informative priors
        with pm.Model() as model:
            # Informative priors from previous posterior
            alpha = pm.LogNormal('alpha', mu=alpha_prior_mu, sigma=alpha_prior_sigma)
            beta = pm.LogNormal('beta', mu=beta_prior_mu, sigma=beta_prior_sigma)

            # Likelihood
            pm.Gamma('obs', alpha=alpha, beta=beta, observed=sample_times)

            # Sample posterior
            try:
                idata = pm.sample(
                    draws=self.mcmc_draws,
                    tune=self.mcmc_tune,
                    chains=self.mcmc_chains,
                    cores=self.mcmc_cores,  # Use multiple cores for parallel sampling
                    target_accept=self.target_accept,
                    progressbar=False,
                    return_inferencedata=True
                )
            except Exception as e:
                print(f"MCMC sampling failed: {e}")
                return None

        return self._process_inference_data(idata)

    def predict_cdf(
        self,
        times: np.ndarray,
        posterior_samples: Dict[str, np.ndarray],
        ci_level: float = 0.70
    ) -> Dict[str, np.ndarray]:
        """
        Compute posterior predictive CDF with credible intervals.

        For each posterior sample of (α, β), compute Gamma CDF.
        Return percentiles across posterior samples.

        Args:
            times: Time points to evaluate CDF
            posterior_samples: Dict with 'alpha' and 'beta' arrays
            ci_level: Credible interval level (default: 0.70)

        Returns:
            Dict with 'lower', 'upper', 'median' CDF arrays
        """
        alpha_samples = posterior_samples['alpha']
        beta_samples = posterior_samples['beta']

        n_samples = len(alpha_samples)
        n_times = len(times)

        # Compute CDF for each posterior sample
        cdf_samples = np.zeros((n_samples, n_times))

        for i in range(n_samples):
            alpha_i = alpha_samples[i]
            beta_i = beta_samples[i]

            # Gamma CDF: F(t; α, β) = P(X ≤ t)
            # scipy uses (shape, scale) = (α, 1/β)
            cdf_samples[i, :] = scipy_gamma.cdf(times, a=alpha_i, scale=1/beta_i)

        # Compute percentiles
        lower_q = (1 - ci_level) / 2
        upper_q = 1 - lower_q

        lower = np.percentile(cdf_samples, lower_q * 100, axis=0)
        upper = np.percentile(cdf_samples, upper_q * 100, axis=0)
        median = np.percentile(cdf_samples, 50, axis=0)

        return {
            'lower': lower,
            'upper': upper,
            'median': median
        }

    def _process_inference_data(
        self,
        idata: az.InferenceData
    ) -> BayesianFitResult:
        """
        Extract summary statistics from InferenceData.

        Computes posterior means, stds, convergence diagnostics.
        """
        # Extract posterior samples
        alpha_samples = idata.posterior['alpha'].values.flatten()
        beta_samples = idata.posterior['beta'].values.flatten()

        # Posterior statistics
        alpha_mean = float(np.mean(alpha_samples))
        beta_mean = float(np.mean(beta_samples))
        alpha_std = float(np.std(alpha_samples))
        beta_std = float(np.std(beta_samples))

        # Convergence diagnostics
        summary = az.summary(idata, var_names=['alpha', 'beta'])

        rhat_alpha = float(summary.loc['alpha', 'r_hat'])
        rhat_beta = float(summary.loc['beta', 'r_hat'])

        ess_alpha = float(summary.loc['alpha', 'ess_bulk'])
        ess_beta = float(summary.loc['beta', 'ess_bulk'])

        # Check convergence (R-hat < 1.01 is good)
        converged = rhat_alpha < 1.01 and rhat_beta < 1.01

        return BayesianFitResult(
            idata=idata,
            alpha_mean=alpha_mean,
            beta_mean=beta_mean,
            alpha_std=alpha_std,
            beta_std=beta_std,
            rhat_alpha=rhat_alpha,
            rhat_beta=rhat_beta,
            ess_alpha=ess_alpha,
            ess_beta=ess_beta,
            converged=converged
        )
