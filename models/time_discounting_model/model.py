"""
Time Discounting Model for Polymarket Event Timing Prediction

This model predicts WHEN an event will happen by analyzing the probability
distribution across all time-based options in a market group.

Model Structure (see mermaid diagram below):
- Hierarchical priors by category (politics, crypto, sports, etc.)
- Volume-based concentration parameters
- Term structure, implied rates, and discount function signals
- Beta-binomial likelihood for resolved events

TODO: ADD SLUG-BASED FEATURES
Coworker is currently parsing slugs for additional labels/factors.
These should be integrated as additional covariates in the model:
- Extract temporal features from slugs (month names, quarters, etc.)
- Category/domain features from event slugs
- Add these as predictors in the event-level mean (μ_event)
"""

import pymc as pm
import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple
import arviz as az


'''
ARCHITECTURE DIAGRAM:

flowchart TD
  H1["μ_cat, σ_cat\ncategory hyperpriors"]:::hyper
  H2["α_vol, β_vol\nvolume hyperprior"]:::hyper

  H1 --> C1
  H1 --> C2
  H1 --> C3

  C1["θ_politics\ncategory mean"]:::cat
  C2["θ_crypto\ncategory mean"]:::cat
  C3["θ_sport\ncategory mean"]:::cat

  C1 --> E
  C2 --> E
  C3 --> E
  H2 --> VOL

  VOL["κ = f(norm volume)\nBeta concentration"]:::signal

  E["μ_event, κ_event\nevent-level Beta params"]:::event

  VOL --> E

  E --> OBS

  TS["term structure\nP at τ₁…τₙ"]:::signal
  IMP["implied rate r*\nbootstrapped from curve"]:::signal
  IMBAL["volume imbalance\n∂vol/∂τ"]:::signal
  DISC["discount fn\nhyperbolic vs exp"]:::signal

  TS --> OBS
  IMP --> OBS
  IMBAL --> OBS
  DISC --> OBS

  OBS["P_obs(τ)\nobserved market price"]:::obs

  OBS --> LIKE["Beta-binomial likelihood\nresolved events"]:::inf

  LIKE --> POST["posterior P(event)\ncredible interval"]:::out
  LIKE --> WAIC["WAIC / LOO-CV\nmodel comparison"]:::out

  classDef hyper  fill:#EEEDFE,stroke:#534AB7,color:#3C3489
  classDef cat    fill:#E1F5EE,stroke:#0F6E56,color:#085041
  classDef event  fill:#FAEEDA,stroke:#854F0B,color:#633806
  classDef signal fill:#F1EFE8,stroke:#5F5E5A,color:#444441
  classDef obs    fill:#F5C4B3,stroke:#993C1D,color:#712B13
  classDef inf    fill:#FAECE7,stroke:#D85A30,color:#993C1D
  classDef out    fill:#9FE1CB,stroke:#0F6E56,color:#085041
  '''


class TimeDiscountingModel:
    """
    Hierarchical Bayesian model for predicting event timing based on market prices.

    Predicts which time option will resolve by modeling:
    - Category-specific baseline rates
    - Volume-based market confidence
    - Term structure features (level, slope, curvature)
    - Implied discount rates
    - Volume imbalance signals
    """

    def __init__(self, discount_function: str = 'hyperbolic'):
        """
        Initialize the model.

        Args:
            discount_function: Type of discount function ('hyperbolic' or 'exponential')
        """
        self.discount_function = discount_function
        self.model = None
        self.trace = None
        self.categories = None

    def prepare_data(self,
                     rates_df: pd.DataFrame,
                     resolved_df: pd.DataFrame) -> Dict:
        """
        Prepare data for PyMC model from data loader output.

        Args:
            rates_df: Output from PolymarketDataLoader.load_full_dataset (price data with rates)
            resolved_df: Resolved outcomes DataFrame

        Returns:
            Dictionary of prepared data arrays for modeling
        """
        # Filter to resolved events only for training
        resolved_event_ids = resolved_df['market_group'].unique()
        df = rates_df[rates_df['market_group'].isin(resolved_event_ids)].copy()

        # Get the latest price observation for each market
        # (or you could aggregate over a time window)
        latest_prices = df.sort_values('date').groupby(['market_group', 'token_id']).last().reset_index()

        # Merge with resolved outcomes
        latest_prices = latest_prices.merge(
            resolved_df[['market_group', 'token_id', 'resolved_outcome']],
            on=['market_group', 'token_id'],
            how='left'
        )

        # Create binary outcome: did this option win?
        latest_prices['won'] = (latest_prices['outcome'] == latest_prices['resolved_outcome']).astype(int)

        # TODO: INTEGRATE SLUG FEATURES HERE
        # When slug parsing is complete, add features like:
        # - temporal_category (from slug parsing: 'Q1', 'January', '2025', etc.)
        # - market_type (from slug parsing)
        # These should be added to latest_prices DataFrame

        # Encode categories
        self.categories = latest_prices['category'].unique()
        category_map = {cat: i for i, cat in enumerate(self.categories)}
        latest_prices['category_idx'] = latest_prices['category'].map(category_map)

        # Encode events
        event_ids = latest_prices['market_group'].unique()
        event_map = {eid: i for i, eid in enumerate(event_ids)}
        latest_prices['event_idx'] = latest_prices['market_group'].map(event_map)

        # Fill NaNs in features with reasonable defaults
        latest_prices['ts_level'] = latest_prices['ts_level'].fillna(0)
        latest_prices['ts_slope'] = latest_prices['ts_slope'].fillna(0)
        latest_prices['ts_curvature'] = latest_prices['ts_curvature'].fillna(0)
        latest_prices['implied_rate'] = latest_prices['implied_rate'].fillna(0)
        latest_prices['volume_num'] = latest_prices['volume_num'].fillna(0)

        # Normalize volume for stability
        volume_normalized = np.log1p(latest_prices['volume_num'])
        volume_normalized = (volume_normalized - volume_normalized.mean()) / (volume_normalized.std() + 1e-6)

        return {
            'prices': latest_prices['price'].values,
            'won': latest_prices['won'].values,
            'category_idx': latest_prices['category_idx'].values,
            'event_idx': latest_prices['event_idx'].values,
            'n_categories': len(self.categories),
            'n_events': len(event_ids),
            'n_obs': len(latest_prices),
            # Signals
            'ts_level': latest_prices['ts_level'].values,
            'ts_slope': latest_prices['ts_slope'].values,
            'ts_curvature': latest_prices['ts_curvature'].values,
            'implied_rate': latest_prices['implied_rate'].values,
            'volume_normalized': volume_normalized.values,
            'time_to_expiration': latest_prices['time_to_expiration'].fillna(0).values,
            # For predictions
            'event_groups': latest_prices['market_group'].values,
            'questions': latest_prices['question'].values
        }

    def build_model(self, data: Dict) -> pm.Model:
        """
        Build the hierarchical PyMC model.

        Args:
            data: Prepared data dictionary from prepare_data()

        Returns:
            PyMC model instance
        """
        with pm.Model() as model:
            # ========================================
            # HYPERPRIORS (Category-level)
            # ========================================
            # Mean and scale for category-specific effects
            μ_cat_hyper = pm.Normal('μ_cat_hyper', mu=0, sigma=2)
            σ_cat_hyper = pm.HalfNormal('σ_cat_hyper', sigma=1)

            # Volume hyperpriors for concentration parameter
            α_vol = pm.HalfNormal('α_vol', sigma=2)
            β_vol = pm.HalfNormal('β_vol', sigma=2)

            # ========================================
            # CATEGORY-LEVEL PARAMETERS
            # ========================================
            θ_category = pm.Normal('θ_category',
                                   mu=μ_cat_hyper,
                                   sigma=σ_cat_hyper,
                                   shape=data['n_categories'])

            # ========================================
            # SIGNAL EFFECTS
            # ========================================
            # Term structure effects
            β_ts_level = pm.Normal('β_ts_level', mu=0, sigma=1)
            β_ts_slope = pm.Normal('β_ts_slope', mu=0, sigma=1)
            β_ts_curvature = pm.Normal('β_ts_curvature', mu=0, sigma=1)

            # Implied rate effect
            β_implied_rate = pm.Normal('β_implied_rate', mu=0, sigma=1)

            # Discount function parameter
            # For hyperbolic: D(t) = 1/(1 + k*t)
            # For exponential: D(t) = exp(-k*t)
            β_discount = pm.HalfNormal('β_discount', sigma=1)

            # TODO: ADD SLUG-BASED FEATURE COEFFICIENTS HERE
            # When slug features are ready, add:
            # β_temporal_category = pm.Normal('β_temporal_category', mu=0, sigma=1)
            # β_market_type = pm.Normal('β_market_type', mu=0, sigma=1)

            # ========================================
            # VOLUME-BASED CONCENTRATION
            # ========================================
            # κ represents how concentrated/confident the Beta distribution is
            # Higher volume -> higher κ -> more concentrated around the mean
            κ_base = pm.Deterministic('κ_base',
                                      α_vol + β_vol * data['volume_normalized'])
            κ = pm.math.exp(κ_base)  # Ensure positive

            # ========================================
            # EVENT-LEVEL PARAMETERS
            # ========================================
            # Each event gets its own mean, informed by category
            μ_event_raw = pm.Normal('μ_event_raw', mu=0, sigma=1, shape=data['n_events'])
            μ_event = pm.Deterministic('μ_event',
                                       θ_category[data['category_idx']] + μ_event_raw[data['event_idx']])

            # ========================================
            # OBSERVATION-LEVEL MEAN (with signals)
            # ========================================
            # Apply discount function
            if self.discount_function == 'hyperbolic':
                discount = 1 / (1 + β_discount * data['time_to_expiration'])
            else:  # exponential
                discount = pm.math.exp(-β_discount * data['time_to_expiration'])

            # Combine all signals into observation mean
            μ_obs = pm.Deterministic('μ_obs',
                pm.math.invlogit(  # Map to [0, 1]
                    μ_event +
                    β_ts_level * data['ts_level'] +
                    β_ts_slope * data['ts_slope'] +
                    β_ts_curvature * data['ts_curvature'] +
                    β_implied_rate * data['implied_rate'] +
                    # TODO: ADD SLUG FEATURES HERE
                    # β_temporal_category * data['temporal_category'] +
                    pm.math.log(discount + 1e-6)  # Discount effect
                )
            )

            # ========================================
            # LIKELIHOOD
            # ========================================
            # Beta-binomial for observed prices and outcomes
            # α and β parameters for Beta distribution
            α_beta = μ_obs * κ
            β_beta = (1 - μ_obs) * κ

            # For resolved events, use actual outcomes
            # Model price as coming from Beta, outcome as Binomial sample
            price_obs = pm.Beta('price_obs',
                               alpha=α_beta,
                               beta=β_beta,
                               observed=data['prices'])

            # Outcome likelihood (did this option win?)
            outcome_obs = pm.Bernoulli('outcome_obs',
                                      p=μ_obs,
                                      observed=data['won'])

        self.model = model
        return model

    def fit(self,
            data: Dict,
            draws: int = 2000,
            tune: int = 1000,
            chains: int = 4,
            target_accept: float = 0.9) -> az.InferenceData:
        """
        Fit the model using MCMC sampling.

        Args:
            data: Prepared data dictionary
            draws: Number of samples to draw
            tune: Number of tuning steps
            chains: Number of MCMC chains
            target_accept: Target acceptance rate

        Returns:
            ArviZ InferenceData object with trace
        """
        if self.model is None:
            self.build_model(data)

        with self.model:
            self.trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=chains,
                target_accept=target_accept,
                return_inferencedata=True
            )

        return self.trace

    def predict(self,
                data: Dict,
                return_samples: bool = False) -> pd.DataFrame:
        """
        Generate predictions for new data.

        Args:
            data: Prepared data dictionary
            return_samples: If True, return full posterior samples

        Returns:
            DataFrame with predictions for each observation
        """
        if self.trace is None:
            raise ValueError("Model must be fit before predicting")

        with self.model:
            ppc = pm.sample_posterior_predictive(
                self.trace,
                var_names=['μ_obs', 'outcome_obs'],
                return_inferencedata=False
            )

        # Get posterior mean and credible intervals
        μ_samples = ppc['μ_obs']  # Shape: (n_samples, n_obs)

        predictions = pd.DataFrame({
            'event_group': data['event_groups'],
            'question': data['questions'],
            'predicted_prob_mean': μ_samples.mean(axis=0),
            'predicted_prob_lower': np.percentile(μ_samples, 2.5, axis=0),
            'predicted_prob_upper': np.percentile(μ_samples, 97.5, axis=0),
            'observed_price': data['prices'],
            'actual_outcome': data['won']
        })

        if return_samples:
            predictions['posterior_samples'] = list(μ_samples.T)

        return predictions

    def compare_models(self, traces: Dict[str, az.InferenceData]) -> pd.DataFrame:
        """
        Compare multiple model variants using WAIC and LOO-CV.

        Args:
            traces: Dictionary of {model_name: trace} for comparison

        Returns:
            DataFrame with model comparison metrics
        """
        comparison = az.compare(traces, ic='waic')
        return comparison