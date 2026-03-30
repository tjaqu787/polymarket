"""
Bayesian Gamma Timing Model

True Bayesian sequential updating for prediction market timing.

Key Features:
- MCMC sampling via PyMC (not MLE)
- Sequential updating: previous posterior → new prior
- Posterior credible intervals (not bootstrap)
- Automatic posterior storage and retrieval

Comparison to Factored Gamma Model:
- Factored: MLE + bootstrap, refits from scratch
- Bayesian: MCMC + sequential, uses previous fit as prior
"""

import pandas as pd
import numpy as np
import arviz as az
from typing import Dict, Optional, List
from dataclasses import dataclass
from datetime import datetime
import re

from .bayesian_gamma_fitter import BayesianGammaFitter, BayesianFitResult
from .posterior_store import PosteriorStore


@dataclass
class FitResult:
    """Result from Bayesian Gamma fit on an event."""
    event_id: str
    fit_date: pd.Timestamp
    alpha_mean: float              # Posterior mean of α
    beta_mean: float               # Posterior mean of β
    alpha_std: float               # Posterior std of α
    beta_std: float                # Posterior std of β
    converged: bool                # MCMC convergence check
    rhat_alpha: float              # R-hat diagnostic
    rhat_beta: float
    credible_intervals: Dict       # Per-market CI bounds
    times: np.ndarray              # Time points used for fit
    cdf_values: np.ndarray         # CDF values used for fit
    is_sequential: bool            # True if used previous posterior as prior


@dataclass
class PredictionResult:
    """Prediction for a specific market."""
    market_id: str
    lower_bound: float
    upper_bound: float
    median: float
    interval_width: float
    alpha_mean: float
    beta_mean: float
    time_to_target: float


class BayesianGammaModel:
    """
    Bayesian Gamma timing model with sequential updating.

    Uses PyMC for MCMC sampling and stores posteriors for sequential updates.
    """

    def __init__(
        self,
        min_buckets: int = 3,
        ci_level: float = 0.70,
        mcmc_draws: int = 500,
        mcmc_tune: int = 500,
        mcmc_chains: int = 2,
        mcmc_cores: int = 4,
        target_accept: float = 0.95,
        posterior_dir: str = "models/bayesian_gamma_model/posteriors"
    ):
        """
        Initialize Bayesian Gamma model.

        Args:
            min_buckets: Minimum term structure points required
            ci_level: Credible interval level (default: 0.70)
            mcmc_draws: MCMC posterior samples per chain
            mcmc_tune: MCMC tuning steps
            mcmc_chains: Number of MCMC chains
            mcmc_cores: Number of CPU cores for parallel MCMC
            target_accept: NUTS target acceptance rate
            posterior_dir: Directory for storing posteriors
        """
        self.min_buckets = min_buckets
        self.ci_level = ci_level

        self.fitter = BayesianGammaFitter(
            mcmc_draws=mcmc_draws,
            mcmc_tune=mcmc_tune,
            mcmc_chains=mcmc_chains,
            mcmc_cores=mcmc_cores,
            target_accept=target_accept
        )

        self.posterior_store = PosteriorStore(base_dir=posterior_dir)

    def fit_event(
        self,
        event_data: pd.DataFrame,
        current_date: pd.Timestamp,
        event_id: str
    ) -> Optional[FitResult]:
        """
        Fit Bayesian Gamma model to event's term structure.

        Uses sequential updating if previous posterior exists:
        1. Check for previous posterior in storage
        2. If exists: fit_sequential() with previous posterior as prior
        3. If not: fit_initial() with default priors
        4. Save new posterior to storage
        5. Return FitResult

        Args:
            event_data: DataFrame with 'No' outcome prices
            current_date: Current date
            event_id: Event identifier

        Returns:
            FitResult or None if fit fails
        """
        # Extract term structure
        term_structure = self._extract_term_structure(event_data, current_date)

        if term_structure is None:
            return None

        times = term_structure['times']
        cdf_values = term_structure['cdf_values']
        market_ids = term_structure['market_ids']

        if len(times) < self.min_buckets:
            return None

        # Check for previous posterior
        prior_result = self.posterior_store.load_latest_posterior(event_id)

        if prior_result is not None:
            # Sequential fit with previous posterior as prior
            prior_date, prior_idata = prior_result
            bayesian_result = self.fitter.fit_sequential(times, cdf_values, prior_idata)
            is_sequential = True
        else:
            # Initial fit with default priors
            bayesian_result = self.fitter.fit_initial(times, cdf_values)
            is_sequential = False

        if bayesian_result is None:
            return None

        # Save new posterior
        try:
            self.posterior_store.save_posterior(event_id, current_date, bayesian_result.idata)
        except Exception as e:
            print(f"Warning: Failed to save posterior for {event_id}: {e}")

        # Compute credible intervals for each market
        posterior_samples = {
            'alpha': bayesian_result.idata.posterior['alpha'].values.flatten(),
            'beta': bayesian_result.idata.posterior['beta'].values.flatten()
        }

        ci_result = self.fitter.predict_cdf(times, posterior_samples, self.ci_level)

        # Map CI to market_ids
        credible_intervals = {}
        for i, (market_id, time) in enumerate(zip(market_ids, times)):
            credible_intervals[market_id] = {
                'lower': ci_result['lower'][i],
                'upper': ci_result['upper'][i],
                'median': ci_result['median'][i],
                'time': time
            }

        return FitResult(
            event_id=event_id,
            fit_date=current_date,
            alpha_mean=bayesian_result.alpha_mean,
            beta_mean=bayesian_result.beta_mean,
            alpha_std=bayesian_result.alpha_std,
            beta_std=bayesian_result.beta_std,
            converged=bayesian_result.converged,
            rhat_alpha=bayesian_result.rhat_alpha,
            rhat_beta=bayesian_result.rhat_beta,
            credible_intervals=credible_intervals,
            times=times,
            cdf_values=cdf_values,
            is_sequential=is_sequential
        )

    def predict(
        self,
        market_id: str,
        fit_result: FitResult,
        current_date: pd.Timestamp
    ) -> Optional[PredictionResult]:
        """
        Get prediction for a specific market.

        Args:
            market_id: Market to predict
            fit_result: FitResult from fit_event()
            current_date: Current date

        Returns:
            PredictionResult or None if market not in fit
        """
        if market_id not in fit_result.credible_intervals:
            return None

        ci = fit_result.credible_intervals[market_id]

        return PredictionResult(
            market_id=market_id,
            lower_bound=ci['lower'],
            upper_bound=ci['upper'],
            median=ci['median'],
            interval_width=ci['upper'] - ci['lower'],
            alpha_mean=fit_result.alpha_mean,
            beta_mean=fit_result.beta_mean,
            time_to_target=ci['time']
        )

    def _extract_term_structure(
        self,
        event_data: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> Optional[Dict]:
        """
        Extract term structure from event data.

        Same logic as factored_gamma_model.
        """
        term_points = []

        for _, row in event_data[event_data['outcome'] == 'No'].iterrows():
            target_date = self._extract_target_date(row['question'])

            if target_date is None:
                continue

            time_to_target = (target_date - current_date).total_seconds() / (365.25 * 24 * 3600)

            if time_to_target <= 0:
                continue

            # Convert No price to Yes price (CDF value)
            yes_price = 1.0 - row['price']
            yes_price = np.clip(yes_price, 0.01, 0.99)

            term_points.append({
                'market_id': row['market_id'],
                'time': time_to_target,
                'cdf_value': yes_price,
                'target_date': target_date
            })

        if len(term_points) == 0:
            return None

        # Sort by time
        term_points.sort(key=lambda x: x['time'])

        times = np.array([p['time'] for p in term_points])
        cdf_values = np.array([p['cdf_value'] for p in term_points])
        market_ids = [p['market_id'] for p in term_points]
        target_dates = [p['target_date'] for p in term_points]

        # Ensure CDF is monotonic
        cdf_values = np.maximum.accumulate(cdf_values)

        return {
            'times': times,
            'cdf_values': cdf_values,
            'market_ids': market_ids,
            'target_dates': target_dates
        }

    def _extract_target_date(self, question: str) -> Optional[datetime]:
        """Extract target date from question text."""
        patterns = [
            r'by ([A-Za-z]+) (\d+),? (\d{4})',
            r'before ([A-Za-z]+) (\d+),? (\d{4})',
            r'no later than ([A-Za-z]+) (\d+),? (\d{4})',
        ]

        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                try:
                    month_name, day, year = match.groups()
                    date_str = f"{year}-{month_name}-{day}"
                    return datetime.strptime(date_str, '%Y-%B-%d')
                except:
                    try:
                        date_str = f"{year}-{month_name}-{day}"
                        return datetime.strptime(date_str, '%Y-%b-%d')
                    except:
                        continue
        return None
