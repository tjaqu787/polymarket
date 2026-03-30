"""
Bayesian Gamma Timing Model

True Bayesian sequential updating for prediction market timing.

Key Components:
- BayesianGammaFitter: MCMC fitting via PyMC
- PosteriorStore: Storage/retrieval of posterior samples
- BayesianGammaModel: Main model orchestrator

Usage:
    from models.bayesian_gamma_model import BayesianGammaModel

    model = BayesianGammaModel(
        min_buckets=3,
        ci_level=0.70,
        mcmc_draws=500
    )

    # Fit event (automatically uses sequential updating if prior exists)
    fit_result = model.fit_event(event_data, current_date, event_id)

    # Get prediction for specific market
    prediction = model.predict(market_id, fit_result, current_date)
"""

from .model import BayesianGammaModel, FitResult, PredictionResult
from .bayesian_gamma_fitter import BayesianGammaFitter, BayesianFitResult
from .posterior_store import PosteriorStore

__all__ = [
    'BayesianGammaModel',
    'FitResult',
    'PredictionResult',
    'BayesianGammaFitter',
    'BayesianFitResult',
    'PosteriorStore',
]
