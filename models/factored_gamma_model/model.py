"""
Factored Gamma Timing Model

Main model class that unifies Gamma CDF fitting with Empirical Bayes factor adjustments.
This is the core orchestrator that composes all model components.

Model Pipeline:
1. fit_factors(): Estimate category priors from historical resolved events (one-time)
2. fit_event(): Fit base Gamma + apply factors for a new event (refitted periodically)
3. predict(): Get CI bounds for specific market within an event
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, List
from dataclasses import dataclass, field
from datetime import datetime
import re

from .gamma_cdf_fitter import GammaCDFFitter
from .empirical_bayes import EmpiricalBayesFactors
from .factor_adjustment import FactorAdjustment


@dataclass
class FitResult:
    """
    Container for event fit results.

    Stores both base and adjusted Gamma parameters, plus credible intervals
    for all markets in the event's term structure.
    """
    event_id: str
    fit_date: pd.Timestamp
    alpha_base: float
    beta_base: float
    alpha_adjusted: float
    beta_adjusted: float
    rmse: float
    aic: float
    credible_intervals: Dict[str, Dict[str, np.ndarray]]  # market_id -> {lower, upper, median, time}
    times: np.ndarray
    cdf_values: np.ndarray
    category: str
    ts_slope: float
    ts_curvature: float
    implied_rate: float


@dataclass
class PredictionResult:
    """
    Container for market-specific prediction.

    Provides CI bounds and interval width for position sizing.
    """
    market_id: str
    lower_bound: float
    upper_bound: float
    median: float
    interval_width: float
    alpha_adjusted: float
    beta_adjusted: float
    time_to_target: float  # Years to target date


class FactoredGammaModel:
    """
    Main Factored Gamma Timing Model.

    Combines:
    - GammaCDFFitter: MLE fitting of Gamma distribution to term structures
    - EmpiricalBayesFactors: Category-level priors from historical data
    - FactorAdjustment: Log-space adjustments to base parameters

    Workflow:
    1. fit_factors() - Estimate factor priors from historical resolved events (once)
    2. fit_event() - Fit base Gamma + apply factors for a new event (periodic)
    3. predict() - Get CI bounds for a specific market (on-demand)
    """

    def __init__(
        self,
        min_buckets: int = 3,
        max_rmse: float = 0.3,
        ci_level: float = 0.70,
        n_bootstrap: int = 500
    ):
        """
        Initialize model with configuration.

        Args:
            min_buckets: Minimum number of term structure points required (default 3)
            max_rmse: Maximum RMSE threshold for accepting fits (default 0.3)
            ci_level: Credible interval level (default 0.70 for 70% CI)
            n_bootstrap: Number of bootstrap samples for CI estimation (default 500)
        """
        self.min_buckets = min_buckets
        self.max_rmse = max_rmse
        self.ci_level = ci_level
        self.n_bootstrap = n_bootstrap

        # Initialize components
        self.fitter = GammaCDFFitter()
        self.eb_factors = EmpiricalBayesFactors()
        self.factors_fitted = False

    def fit_factors(
        self,
        resolved_events_df: pd.DataFrame,
        holdout_end_date: str
    ) -> None:
        """
        Fit empirical Bayes factors from historical resolved events.

        This should be called once before backtesting, using data up to
        holdout_end_date. The fitted factors are then treated as fixed
        priors during the backtest period.

        Args:
            resolved_events_df: Resolved events with term structure data
                Expected columns:
                    - event_id or semantic_group_id
                    - category
                    - resolution_date
                    - date (observation date)
                    - times (array of tenors in years)
                    - cdf_values (array of Yes prices)
                    - ts_slope, ts_curvature, implied_rate
            holdout_end_date: Cutoff date for factor estimation (e.g., "2025-10-05")
                Only events with date <= holdout_end_date are used

        Side Effects:
            - Populates self.eb_factors with fitted factor parameters
            - Sets self.factors_fitted = True
            - Can optionally save factors to disk for reuse
        """
        print(f"\nFitting Empirical Bayes factors (holdout end: {holdout_end_date})...")

        # Filter to holdout period
        if 'date' in resolved_events_df.columns:
            resolved_events_df['date'] = pd.to_datetime(resolved_events_df['date'])
            holdout_date = pd.to_datetime(holdout_end_date)
            filtered_df = resolved_events_df[resolved_events_df['date'] <= holdout_date].copy()
        else:
            filtered_df = resolved_events_df.copy()

        print(f"Filtered to {len(filtered_df)} observations in holdout period")

        # Fit factors
        self.eb_factors.fit(filtered_df, self.fitter)
        self.factors_fitted = True

        print("✓ Empirical Bayes factors fitted successfully")

    def fit_event(
        self,
        event_data: pd.DataFrame,
        current_date: pd.Timestamp,
        event_id: str
    ) -> Optional[FitResult]:
        """
        Fit Gamma model to a single event's term structure.

        Args:
            event_data: DataFrame with 'No' outcome prices for one event
                Expected columns:
                    - market_id
                    - outcome ('No')
                    - price
                    - question (to parse target dates)
                    - resolution_date or end_date
                    - category
                    - ts_slope, ts_curvature, implied_rate (optional)
            current_date: Current backtest date
            event_id: Event identifier (semantic_group_id or event_id)

        Returns:
            FitResult object or None if fit failed

        Implementation:
            1. Extract term structure from event_data
            2. Validate (≥ min_buckets points, times strictly increasing)
            3. Fit base Gamma via GammaCDFFitter (MLE, refits from scratch)
            4. Apply factor adjustments via FactorAdjustment
            5. Compute credible intervals using ADJUSTED parameters
            6. Return FitResult with all parameters and CI bounds

        Note on Refitting:
            Each call refits from scratch via MLE (frequentist approach).
            This is NOT Bayesian sequential updating:
            - Estimates (α, β) via maximum likelihood on current data
            - Does NOT use previous fit as prior
            - Bootstrap provides frequentist confidence intervals

            For true Bayesian refitting: use previous posterior as new prior,
            update via MCMC (PyMC), get credible intervals from posterior samples.
        """
        # Extract term structure
        try:
            term_structure = self._extract_term_structure(event_data, current_date)
        except Exception as e:
            print(f"Failed to extract term structure for {event_id}: {e}")
            return None

        if term_structure is None:
            return None

        times = term_structure['times']
        cdf_values = term_structure['cdf_values']
        market_ids = term_structure['market_ids']
        target_dates = term_structure['target_dates']

        # Validate
        if len(times) < self.min_buckets:
            return None

        # Fit base Gamma
        try:
            fit_result = self.fitter.fit(times, cdf_values)
        except Exception as e:
            print(f"Gamma fit failed for {event_id}: {e}")
            return None

        # Check fit quality
        if fit_result['rmse'] > self.max_rmse:
            return None

        alpha_base = fit_result['alpha']
        beta_base = fit_result['beta']
        rmse = fit_result['rmse']
        aic = fit_result['aic']

        # Get category and features (with defaults)
        category = event_data.iloc[0].get('category', 'unknown')
        ts_slope = event_data.iloc[0].get('ts_slope', 0.0)
        ts_curvature = event_data.iloc[0].get('ts_curvature', 0.0)
        implied_rate = event_data.iloc[0].get('implied_rate', 0.0)

        # Apply factor adjustments
        if self.factors_fitted:
            try:
                alpha_adjusted, beta_adjusted = FactorAdjustment.adjust(
                    alpha_base=alpha_base,
                    beta_base=beta_base,
                    category=category,
                    ts_slope=ts_slope,
                    ts_curvature=ts_curvature,
                    implied_rate=implied_rate,
                    eb_factors=self.eb_factors
                )
            except Exception as e:
                print(f"Factor adjustment failed for {event_id}: {e}")
                # Fall back to base parameters
                alpha_adjusted = alpha_base
                beta_adjusted = beta_base
        else:
            # No factors fitted yet, use base params
            alpha_adjusted = alpha_base
            beta_adjusted = beta_base

        # Compute credible intervals by bootstrapping the entire fit + adjustment pipeline
        # This gives real uncertainty that reflects data quality, not arbitrary noise
        try:
            ci_result = self._compute_adjusted_ci(
                times=times,
                cdf_values=cdf_values,
                alpha_adjusted=alpha_adjusted,
                beta_adjusted=beta_adjusted,
                category=category,
                ts_slope=ts_slope,
                ts_curvature=ts_curvature,
                implied_rate=implied_rate
            )

            # Map CI bounds to market_ids
            credible_intervals = {}
            for i, (market_id, time, target_date) in enumerate(zip(market_ids, times, target_dates)):
                credible_intervals[market_id] = {
                    'lower': ci_result['lower'][i],
                    'upper': ci_result['upper'][i],
                    'median': ci_result['median'][i],
                    'time': time,
                    'target_date': target_date
                }

        except Exception as e:
            print(f"CI computation failed for {event_id}: {e}")
            return None

        # Return FitResult
        return FitResult(
            event_id=event_id,
            fit_date=current_date,
            alpha_base=alpha_base,
            beta_base=beta_base,
            alpha_adjusted=alpha_adjusted,
            beta_adjusted=beta_adjusted,
            rmse=rmse,
            aic=aic,
            credible_intervals=credible_intervals,
            times=times,
            cdf_values=cdf_values,
            category=category,
            ts_slope=ts_slope,
            ts_curvature=ts_curvature,
            implied_rate=implied_rate
        )

    def predict(
        self,
        market_id: str,
        fit_result: FitResult,
        current_date: pd.Timestamp
    ) -> Optional[PredictionResult]:
        """
        Get prediction for a specific market from a fitted event.

        Args:
            market_id: Market to predict
            fit_result: FitResult from fit_event()
            current_date: Current date (for recalculating time to target)

        Returns:
            PredictionResult with CI bounds and interval width

        Edge Cases:
            - market_id not in fit_result: return None
        """
        if market_id not in fit_result.credible_intervals:
            return None

        ci = fit_result.credible_intervals[market_id]

        lower_bound = ci['lower']
        upper_bound = ci['upper']
        median = ci['median']
        interval_width = upper_bound - lower_bound
        time_to_target = ci['time']

        return PredictionResult(
            market_id=market_id,
            lower_bound=lower_bound,
            upper_bound=upper_bound,
            median=median,
            interval_width=interval_width,
            alpha_adjusted=fit_result.alpha_adjusted,
            beta_adjusted=fit_result.beta_adjusted,
            time_to_target=time_to_target
        )

    def _extract_term_structure(
        self,
        event_data: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> Optional[Dict]:
        """
        Extract term structure from event data.

        Uses resolution_date (or end_date) from data, calculates
        times to expiry, converts No prices to Yes prices, and sorts.

        Returns:
            Dict with times, cdf_values, market_ids, target_dates
            or None if extraction fails
        """
        term_structure_data = []

        for _, row in event_data.iterrows():
            # Use resolution_date or end_date from data
            if 'resolution_date' in row.index and pd.notna(row['resolution_date']):
                target_date = pd.to_datetime(row['resolution_date'])
            elif 'end_date' in row.index and pd.notna(row['end_date']):
                target_date = pd.to_datetime(row['end_date'])
            else:
                # Fallback: try parsing from question
                target_date = self._extract_target_date(row['question'])
                if target_date is None:
                    continue

            # Calculate time to target in years
            time_to_target = (target_date - current_date).days / 365.25
            if time_to_target <= 0:
                continue  # Skip expired markets

            # Convert No price to Yes price
            # Since P(No) = 1 - P(Yes), we have P(Yes) = 1 - P(No)
            no_price = row['price']
            yes_price = 1.0 - no_price

            term_structure_data.append({
                'market_id': row['market_id'],
                'time': time_to_target,
                'cdf_value': yes_price,
                'target_date': target_date
            })

        if len(term_structure_data) == 0:
            return None

        # Sort by time
        term_structure_data = sorted(term_structure_data, key=lambda x: x['time'])

        # Extract arrays
        times = np.array([x['time'] for x in term_structure_data])
        cdf_values = np.array([x['cdf_value'] for x in term_structure_data])
        market_ids = [x['market_id'] for x in term_structure_data]
        target_dates = [x['target_date'] for x in term_structure_data]

        # Remove duplicates (keep first occurrence)
        unique_times = []
        unique_cdf_values = []
        unique_market_ids = []
        unique_target_dates = []

        for i, t in enumerate(times):
            if i == 0 or t != times[i-1]:
                unique_times.append(t)
                unique_cdf_values.append(cdf_values[i])
                unique_market_ids.append(market_ids[i])
                unique_target_dates.append(target_dates[i])

        return {
            'times': np.array(unique_times),
            'cdf_values': np.array(unique_cdf_values),
            'market_ids': unique_market_ids,
            'target_dates': unique_target_dates
        }

    def _extract_target_date(self, question: str) -> Optional[datetime]:
        """
        Extract target date from question text.

        Parses patterns like:
        - "by March 15, 2026"
        - "before April 1, 2026"
        - "no later than June 30, 2026"

        Returns:
            datetime object or None if parsing fails
        """
        patterns = [
            r'by ([A-Za-z]+) (\d+),? (\d{4})',  # "by March 15, 2026" or "by March 15 2026"
            r'before ([A-Za-z]+) (\d+),? (\d{4})',  # "before March 15, 2026"
            r'no later than ([A-Za-z]+) (\d+),? (\d{4})',  # "no later than March 15, 2026"
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
                        # Try abbreviated month
                        date_str = f"{year}-{month_name}-{day}"
                        return datetime.strptime(date_str, '%Y-%b-%d')
                    except:
                        continue
        return None

    def _compute_adjusted_ci(
        self,
        times: np.ndarray,
        cdf_values: np.ndarray,
        alpha_adjusted: float,
        beta_adjusted: float,
        category: str,
        ts_slope: float,
        ts_curvature: float,
        implied_rate: float
    ) -> Dict:
        """
        Compute credible intervals by bootstrapping the entire fit + adjustment pipeline.

        This is the CORRECT implementation: for each bootstrap iteration,
        we resample from the implied PDF, fit base Gamma, apply factor adjustments,
        and predict CDF. This gives real uncertainty that reflects data quality and fit.

        Args:
            times: Time points
            cdf_values: Original CDF values
            alpha_adjusted: Adjusted shape parameter (for reference, not used)
            beta_adjusted: Adjusted rate parameter (for reference, not used)
            category: Event category for factor adjustments
            ts_slope: Term structure slope
            ts_curvature: Term structure curvature
            implied_rate: Implied discount rate

        Returns:
            Dict with lower, upper, median CI bounds
        """
        # Convert CDF to PDF for resampling
        time_midpoints, pdf_values = self.fitter.cdf_to_pdf(times, cdf_values)

        bootstrap_cdfs = []
        n_samples = 10000  # Samples per bootstrap iteration

        for _ in range(self.n_bootstrap):
            try:
                # Resample from implied PDF
                sample_times = np.random.choice(
                    time_midpoints,
                    size=n_samples,
                    replace=True,
                    p=pdf_values / pdf_values.sum()
                )

                # Fit base Gamma to bootstrap sample
                from scipy.stats import gamma
                shape, loc, scale = gamma.fit(sample_times, floc=0)
                alpha_base_boot = shape
                beta_base_boot = 1 / scale

                # Apply factor adjustments if factors are fitted
                if self.factors_fitted:
                    alpha_adj_boot, beta_adj_boot = FactorAdjustment.adjust(
                        alpha_base=alpha_base_boot,
                        beta_base=beta_base_boot,
                        category=category,
                        ts_slope=ts_slope,
                        ts_curvature=ts_curvature,
                        implied_rate=implied_rate,
                        eb_factors=self.eb_factors
                    )
                else:
                    alpha_adj_boot = alpha_base_boot
                    beta_adj_boot = beta_base_boot

                # Predict CDF using adjusted parameters
                boot_cdf = self.fitter.predict_cdf(times, alpha_adj_boot, beta_adj_boot)
                bootstrap_cdfs.append(boot_cdf)

            except:
                # Skip failed bootstrap iterations
                continue

        # Check if we have enough successful iterations
        if len(bootstrap_cdfs) < self.n_bootstrap / 2:
            # Fallback: use point estimates if bootstrap failed
            point_cdf = self.fitter.predict_cdf(times, alpha_adjusted, beta_adjusted)
            return {
                'lower': point_cdf,
                'upper': point_cdf,
                'median': point_cdf
            }

        bootstrap_cdfs = np.array(bootstrap_cdfs)

        # Calculate percentiles
        alpha = (1 - self.ci_level) / 2
        lower_percentile = alpha * 100
        upper_percentile = (1 - alpha) * 100

        return {
            'lower': np.percentile(bootstrap_cdfs, lower_percentile, axis=0),
            'upper': np.percentile(bootstrap_cdfs, upper_percentile, axis=0),
            'median': np.percentile(bootstrap_cdfs, 50, axis=0)
        }
