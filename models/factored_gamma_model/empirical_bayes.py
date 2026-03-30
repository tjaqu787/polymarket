"""
Empirical Bayes Factor Estimation

Estimates category-specific factor priors from historical resolved events.
This implements the "Empirical Bayes" layer: learn hyperparameters from data,
then treat them as fixed priors for new events during backtesting.

Factor Structure (from instructions.txt):
    log_α = log(α_base) + θ_shape[category] + β_curv * curvature
    log_β = log(β_base) + φ_rate[category] + β_slope * slope + β_rate * implied_rate
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, Optional, List
from sklearn.linear_model import LinearRegression
from collections import defaultdict


class EmpiricalBayesFactors:
    """
    Estimate factor parameters from historical resolved events.

    Workflow:
    1. For each resolved event, fit base Gamma(α_base, β_base) to term structure
    2. Aggregate by category and regress in log space:
        - log(α_base) ~ category + ts_curvature
        - log(β_base) ~ category + ts_slope + implied_rate
    3. Store fitted coefficients for use as fixed priors during backtest

    This is "Empirical Bayes" because we:
    - Estimate hyperparameters (θ_shape, φ_rate, β coefficients) from data
    - Treat them as fixed priors (no further updates during backtest)
    - Avoid full Bayesian inference (no MCMC on hyperparameters)
    """

    def __init__(self):
        """Initialize empty factors."""
        self.factors = None
        self.categories = None
        self.fitted = False

    def fit(self,
            resolved_events_df: pd.DataFrame,
            fitter: 'GammaCDFFitter',
            min_events_per_category: int = 5) -> None:
        """
        Fit factor parameters from resolved events.

        Args:
            resolved_events_df: DataFrame with columns:
                - event_id or semantic_group_id: Event identifier
                - category: Event category (politics, crypto, sports, etc.)
                - resolution_date: Actual resolution date
                - date: Observation date (for calculating time_to_resolution)
                - times: Array of tenors in years [0.25, 0.5, 1.0, ...]
                - cdf_values: Array of Yes prices at each tenor
                - ts_slope: Term structure slope
                - ts_curvature: Term structure curvature
                - implied_rate: Bootstrapped implied rate
                (Note: Some columns may need to be computed if not present)

            fitter: GammaCDFFitter instance for fitting base parameters
            min_events_per_category: Minimum events required per category (default 5)

        Implementation Steps:
            1. For each resolved event:
                - Fit base Gamma(α_base, β_base) to term structure
                - Store: (α_base, β_base, category, ts_slope, ts_curvature, implied_rate)
            2. Aggregate by category:
                - Regress log(α_base) ~ category + ts_curvature
                - Regress log(β_base) ~ category + ts_slope + implied_rate
            3. Store fitted coefficients in self.factors dict

        Note:
            Events are expected to be pre-filtered to date ≤ eb_holdout_end_date
            by the caller (FactoredGammaModel.fit_factors() or strategy initialization).
        """
        print(f"Fitting Empirical Bayes factors from {len(resolved_events_df)} events...")

        # Storage for fitted base parameters
        fit_data = []

        # Group by event
        event_col = 'semantic_group_id' if 'semantic_group_id' in resolved_events_df.columns else 'event_id'

        for event_id, event_group in resolved_events_df.groupby(event_col):
            # Take most recent observation (or could take ~30 days before resolution)
            # For simplicity, take last observation
            event_row = event_group.iloc[-1]

            # Extract term structure
            if 'times' not in event_row or 'cdf_values' not in event_row:
                # Skip if term structure data missing
                continue

            times = event_row['times']
            cdf_values = event_row['cdf_values']

            # Validate term structure
            if not isinstance(times, (list, np.ndarray)) or len(times) < 3:
                continue
            if not isinstance(cdf_values, (list, np.ndarray)) or len(cdf_values) < 3:
                continue

            times = np.array(times)
            cdf_values = np.array(cdf_values)

            # Fit base Gamma
            try:
                fit_result = fitter.fit(times, cdf_values)
                alpha_base = fit_result['alpha']
                beta_base = fit_result['beta']
                rmse = fit_result['rmse']

                # Filter poor fits
                if rmse > 0.3:  # max_rmse threshold
                    continue

                # Store fit data
                fit_data.append({
                    'event_id': event_id,
                    'alpha_base': alpha_base,
                    'beta_base': beta_base,
                    'log_alpha_base': np.log(alpha_base),
                    'log_beta_base': np.log(beta_base),
                    'category': event_row.get('category', 'unknown'),
                    'ts_slope': event_row.get('ts_slope', 0.0),
                    'ts_curvature': event_row.get('ts_curvature', 0.0),
                    'implied_rate': event_row.get('implied_rate', 0.0),
                    'rmse': rmse
                })

            except Exception as e:
                # Skip failed fits
                continue

        if len(fit_data) == 0:
            raise ValueError("No successful fits found in resolved events data")

        # Convert to DataFrame
        fit_df = pd.DataFrame(fit_data)
        print(f"Successfully fit {len(fit_df)} events")

        # Handle missing features
        fit_df['ts_slope'].fillna(0.0, inplace=True)
        fit_df['ts_curvature'].fillna(0.0, inplace=True)
        fit_df['implied_rate'].fillna(0.0, inplace=True)

        # Get unique categories
        category_counts = fit_df['category'].value_counts()
        print(f"\nCategory distribution:")
        print(category_counts)

        # Filter categories with too few events
        valid_categories = category_counts[category_counts >= min_events_per_category].index.tolist()
        print(f"\nCategories with >= {min_events_per_category} events: {valid_categories}")

        # Mark rare categories as 'other'
        fit_df['category_grouped'] = fit_df['category'].apply(
            lambda x: x if x in valid_categories else 'other'
        )

        # Create dummy variables for categories
        category_dummies_alpha = pd.get_dummies(fit_df['category_grouped'], prefix='cat', drop_first=True)
        category_dummies_beta = pd.get_dummies(fit_df['category_grouped'], prefix='cat', drop_first=True)

        # Prepare features for alpha regression
        X_alpha = category_dummies_alpha.copy()
        X_alpha['ts_curvature'] = fit_df['ts_curvature'].values
        y_alpha = fit_df['log_alpha_base'].values

        # Prepare features for beta regression
        X_beta = category_dummies_beta.copy()
        X_beta['ts_slope'] = fit_df['ts_slope'].values
        X_beta['implied_rate'] = fit_df['implied_rate'].values
        y_beta = fit_df['log_beta_base'].values

        # Fit regressions
        reg_alpha = LinearRegression(fit_intercept=True)
        reg_beta = LinearRegression(fit_intercept=True)

        reg_alpha.fit(X_alpha, y_alpha)
        reg_beta.fit(X_beta, y_beta)

        print(f"\nAlpha regression R²: {reg_alpha.score(X_alpha, y_alpha):.3f}")
        print(f"Beta regression R²: {reg_beta.score(X_beta, y_beta):.3f}")

        # Extract coefficients
        # Category effects (relative to reference category)
        categories_in_model = [col.replace('cat_', '') for col in category_dummies_alpha.columns]
        reference_category = [cat for cat in fit_df['category_grouped'].unique()
                            if cat not in categories_in_model][0]

        theta_shape = {reference_category: 0.0}  # Reference category has 0 adjustment
        phi_rate = {reference_category: 0.0}

        for i, cat in enumerate(categories_in_model):
            theta_shape[cat] = reg_alpha.coef_[i]
            phi_rate[cat] = reg_beta.coef_[i]

        # Global coefficients (last columns in X matrices)
        beta_curvature = reg_alpha.coef_[-1]  # Last coef is ts_curvature
        beta_slope = reg_beta.coef_[-2]  # Second to last is ts_slope
        beta_implied_rate = reg_beta.coef_[-1]  # Last is implied_rate

        # Store factors
        self.factors = {
            'theta_shape': theta_shape,
            'phi_rate': phi_rate,
            'beta_curvature': beta_curvature,
            'beta_slope': beta_slope,
            'beta_implied_rate': beta_implied_rate,
            'intercept_alpha': reg_alpha.intercept_,
            'intercept_beta': reg_beta.intercept_,
            'reference_category': reference_category,
            'categories': list(theta_shape.keys()),
            'n_events_fitted': len(fit_df)
        }

        self.categories = list(theta_shape.keys())
        self.fitted = True

        print(f"\nFitted factors:")
        print(f"  Reference category: {reference_category}")
        print(f"  θ_shape (category adjustments for α):")
        for cat, val in theta_shape.items():
            print(f"    {cat}: {val:.3f}")
        print(f"  φ_rate (category adjustments for β):")
        for cat, val in phi_rate.items():
            print(f"    {cat}: {val:.3f}")
        print(f"  β_curvature (global curvature effect on α): {beta_curvature:.3f}")
        print(f"  β_slope (global slope effect on β): {beta_slope:.3f}")
        print(f"  β_implied_rate (global implied rate effect on β): {beta_implied_rate:.3f}")

    def get_factor_adjustments(
        self,
        category: str,
        ts_slope: float,
        ts_curvature: float,
        implied_rate: float
    ) -> Dict[str, float]:
        """
        Get factor adjustments for a given event.

        Args:
            category: Event category
            ts_slope: Term structure slope
            ts_curvature: Term structure curvature
            implied_rate: Implied discount rate

        Returns:
            Dictionary with:
                - 'log_alpha_adjustment': Total adjustment to log(α_base)
                - 'log_beta_adjustment': Total adjustment to log(β_base)

        Implementation:
            log_α_adj = θ_shape[category] + β_curvature * ts_curvature
            log_β_adj = φ_rate[category] + β_slope * ts_slope + β_implied_rate * implied_rate

        Edge Cases:
            - Unknown category: use reference category (0 adjustment)
            - Missing features (NaN): treat as 0
        """
        if not self.fitted:
            # No factors fitted yet, return 0 adjustments (no effect)
            return {
                'log_alpha_adjustment': 0.0,
                'log_beta_adjustment': 0.0
            }

        # Handle unknown category
        if category not in self.factors['theta_shape']:
            # Use reference category (0 adjustment)
            category = self.factors['reference_category']

        # Handle NaN features
        ts_slope = 0.0 if np.isnan(ts_slope) else ts_slope
        ts_curvature = 0.0 if np.isnan(ts_curvature) else ts_curvature
        implied_rate = 0.0 if np.isnan(implied_rate) else implied_rate

        # Calculate adjustments
        log_alpha_adjustment = (
            self.factors['theta_shape'][category] +
            self.factors['beta_curvature'] * ts_curvature
        )

        log_beta_adjustment = (
            self.factors['phi_rate'][category] +
            self.factors['beta_slope'] * ts_slope +
            self.factors['beta_implied_rate'] * implied_rate
        )

        return {
            'log_alpha_adjustment': log_alpha_adjustment,
            'log_beta_adjustment': log_beta_adjustment
        }

    def save(self, path: str) -> None:
        """
        Save fitted factors to JSON file.

        Args:
            path: File path to save factors (e.g., 'factors.json')
        """
        if not self.fitted:
            raise ValueError("No factors to save - call fit() first")

        with open(path, 'w') as f:
            json.dump(self.factors, f, indent=2)

        print(f"Saved factors to {path}")

    def load(self, path: str) -> None:
        """
        Load fitted factors from JSON file.

        Args:
            path: File path to load factors from
        """
        with open(path, 'r') as f:
            self.factors = json.load(f)

        self.categories = self.factors['categories']
        self.fitted = True

        print(f"Loaded factors from {path}")
        print(f"  Categories: {self.categories}")
        print(f"  N events: {self.factors['n_events_fitted']}")
