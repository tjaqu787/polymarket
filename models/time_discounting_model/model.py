"""
Time Discounting Model for Polymarket Event Timing Prediction

This model predicts WHEN an event will happen by analyzing the probability
distribution across all time-based options in a market group.

Model Structure (see mermaid diagram below):
- Hierarchical priors by category (politics, crypto, sports, etc.)
- Volume-based concentration parameters
- Term structure, implied rates, and discount function signals
- Slug-based cooccurrence features (token counts, document frequencies, cooccurrence patterns)
- Beta-binomial likelihood for resolved events

SLUG-BASED FEATURES INTEGRATED:
- Token count (number of unique tokens in market slug/question)
- Average token document frequency (how common are the tokens across all markets)
- Max cooccurrence (strength of token pair relationships)
- Token diversity (ratio of unique to total tokens)
These are loaded from timing_text_features and timing_token_cooccurrence tables.
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
        # Use event_id from rates_df, market_group from resolved_df
        resolved_event_ids = resolved_df['market_group'].unique()
        event_col = 'event_id' if 'event_id' in rates_df.columns else 'market_group'
        df = rates_df[rates_df[event_col].isin(resolved_event_ids)].copy()

        # Get the latest price observation for each market
        # (or you could aggregate over a time window)
        group_col = 'event_id' if 'event_id' in df.columns else 'market_group'
        latest_prices = df.sort_values('date').groupby([group_col, 'token_id']).last().reset_index()

        # Merge with resolved outcomes
        # Merge on market_id (resolved_df has market-level outcomes, not token-level)
        latest_prices = latest_prices.merge(
            resolved_df[['market_id', 'resolved_outcome']],
            on='market_id',
            how='left'
        )

        # Create binary outcome: did this option win?
        # Handle outcome_x/outcome_y from merges
        outcome_col = 'outcome' if 'outcome' in latest_prices.columns else ('outcome_x' if 'outcome_x' in latest_prices.columns else 'outcome_y')
        latest_prices['won'] = (latest_prices[outcome_col] == latest_prices['resolved_outcome']).astype(int)

        # Encode categories
        self.categories = latest_prices['category'].unique()
        category_map = {cat: i for i, cat in enumerate(self.categories)}
        latest_prices['category_idx'] = latest_prices['category'].map(category_map)

        # Encode events - use the group column (event_id or market_group)
        event_ids_col = latest_prices[group_col].unique()
        event_map = {eid: i for i, eid in enumerate(event_ids_col)}
        latest_prices['event_idx'] = latest_prices[group_col].map(event_map)

        # Fill NaNs in features with reasonable defaults
        latest_prices['ts_level'] = latest_prices['ts_level'].fillna(0)
        latest_prices['ts_slope'] = latest_prices['ts_slope'].fillna(0)
        latest_prices['ts_curvature'] = latest_prices['ts_curvature'].fillna(0)
        latest_prices['implied_rate'] = latest_prices['implied_rate'].fillna(0)
        latest_prices['volume_num'] = latest_prices['volume_num'].fillna(0)

        # Fill NaNs in cooccurrence features
        latest_prices['token_count'] = latest_prices['token_count'].fillna(0)
        latest_prices['avg_token_df'] = latest_prices['avg_token_df'].fillna(0)
        latest_prices['max_token_df'] = latest_prices['max_token_df'].fillna(0)
        latest_prices['token_diversity'] = latest_prices['token_diversity'].fillna(0)
        latest_prices['num_pairs'] = latest_prices['num_pairs'].fillna(0)
        latest_prices['avg_cooccurrence'] = latest_prices['avg_cooccurrence'].fillna(0)
        latest_prices['max_cooccurrence'] = latest_prices['max_cooccurrence'].fillna(0)

        # Normalize volume for stability
        volume_normalized = np.log1p(latest_prices['volume_num'])
        volume_normalized = (volume_normalized - volume_normalized.mean()) / (volume_normalized.std() + 1e-6)

        # Normalize cooccurrence features
        token_count_norm = np.log1p(latest_prices['token_count'])
        token_count_norm = (token_count_norm - token_count_norm.mean()) / (token_count_norm.std() + 1e-6)

        avg_token_df_norm = np.log1p(latest_prices['avg_token_df'])
        avg_token_df_norm = (avg_token_df_norm - avg_token_df_norm.mean()) / (avg_token_df_norm.std() + 1e-6)

        max_cooccurrence_norm = np.log1p(latest_prices['max_cooccurrence'])
        max_cooccurrence_norm = (max_cooccurrence_norm - max_cooccurrence_norm.mean()) / (max_cooccurrence_norm.std() + 1e-6)

        # Clip prices to be strictly in (0, 1) for Beta distribution
        # Beta distribution requires values strictly between 0 and 1
        epsilon = 1e-6
        prices_clipped = np.clip(latest_prices['price'].values, epsilon, 1 - epsilon)

        return {
            'prices': prices_clipped,
            'won': latest_prices['won'].values,
            'category_idx': latest_prices['category_idx'].values,
            'event_idx': latest_prices['event_idx'].values,
            'n_categories': len(self.categories),
            'n_events': len(event_ids_col),
            'n_obs': len(latest_prices),
            # Signals
            'ts_level': latest_prices['ts_level'].values,
            'ts_slope': latest_prices['ts_slope'].values,
            'ts_curvature': latest_prices['ts_curvature'].values,
            'implied_rate': latest_prices['implied_rate'].values,
            'volume_normalized': volume_normalized.values,
            'time_to_expiration': latest_prices['time_to_expiration'].fillna(0).values,
            # Cooccurrence features (SLUG-BASED)
            'token_count_norm': token_count_norm.values,
            'avg_token_df_norm': avg_token_df_norm.values,
            'max_cooccurrence_norm': max_cooccurrence_norm.values,
            'token_diversity': latest_prices['token_diversity'].values,
            # For predictions
            'event_groups': latest_prices[group_col].values,
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

            # SLUG-BASED COOCCURRENCE FEATURE COEFFICIENTS
            β_token_count = pm.Normal('β_token_count', mu=0, sigma=1)
            β_avg_token_df = pm.Normal('β_avg_token_df', mu=0, sigma=1)
            β_max_cooccurrence = pm.Normal('β_max_cooccurrence', mu=0, sigma=1)
            β_token_diversity = pm.Normal('β_token_diversity', mu=0, sigma=1)

            # ========================================
            # VOLUME-BASED CONCENTRATION
            # ========================================
            # κ represents how concentrated/confident the Beta distribution is
            # Higher volume -> higher κ -> more concentrated around the mean
            κ_base = pm.Deterministic('κ_base',
                                      α_vol + β_vol * data['volume_normalized'])
            # Clip κ_base to prevent overflow, then exp to ensure positive
            # Keep κ in reasonable range [1, 100] to avoid numerical issues
            κ = pm.math.clip(pm.math.exp(κ_base), 1.0, 100.0)

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
            # Clip to keep away from extreme values that cause numerical issues
            μ_obs_raw = pm.math.invlogit(  # Map to [0, 1]
                μ_event +
                β_ts_level * data['ts_level'] +
                β_ts_slope * data['ts_slope'] +
                β_ts_curvature * data['ts_curvature'] +
                β_implied_rate * data['implied_rate'] +
                # SLUG-BASED COOCCURRENCE FEATURES
                β_token_count * data['token_count_norm'] +
                β_avg_token_df * data['avg_token_df_norm'] +
                β_max_cooccurrence * data['max_cooccurrence_norm'] +
                β_token_diversity * data['token_diversity'] +
                pm.math.log(discount + 1e-6)  # Discount effect
            )
            # Ensure μ_obs is strictly in (0, 1) for Beta distribution stability
            μ_obs = pm.Deterministic('μ_obs',
                                     pm.math.clip(μ_obs_raw, 1e-6, 1 - 1e-6))

            # ========================================
            # LIKELIHOOD
            # ========================================
            # Beta-binomial for observed prices and outcomes
            # α and β parameters for Beta distribution
            # Ensure parameters are valid (> 0) and not too extreme
            α_beta = pm.math.maximum(μ_obs * κ, 0.1)  # Ensure α > 0
            β_beta = pm.math.maximum((1 - μ_obs) * κ, 0.1)  # Ensure β > 0

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