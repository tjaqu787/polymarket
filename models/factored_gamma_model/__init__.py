"""
Factored Gamma Timing Model Package

A unified prediction market timing model that combines Gamma CDF fitting
with Empirical Bayes factor adjustments for category-level learning.

Components:
- GammaCDFFitter: Fit Gamma distributions to term structure CDFs via MLE
- EmpiricalBayesFactors: Estimate category-specific priors from historical data
- FactorAdjustment: Apply log-space adjustments to base parameters
- FactoredGammaModel: Main model orchestrator

Usage:
    from models.factored_gamma_model import FactoredGammaModel

    model = FactoredGammaModel(
        min_buckets=3,
        max_rmse=0.3,
        ci_level=0.70,
        n_bootstrap=500
    )

    # Fit empirical Bayes factors (once, before backtest)
    model.fit_factors(resolved_events_df, holdout_end_date="2025-10-05")

    # Fit event (periodic, during backtest)
    fit_result = model.fit_event(event_data, current_date, event_id)

    # Get prediction for specific market
    prediction = model.predict(market_id, fit_result, current_date)
"""

from .gamma_cdf_fitter import GammaCDFFitter
from .empirical_bayes import EmpiricalBayesFactors
from .factor_adjustment import FactorAdjustment
from .model import FactoredGammaModel, FitResult, PredictionResult

__all__ = [
    'GammaCDFFitter',
    'EmpiricalBayesFactors',
    'FactorAdjustment',
    'FactoredGammaModel',
    'FitResult',
    'PredictionResult'
]

__version__ = '1.0.0'
