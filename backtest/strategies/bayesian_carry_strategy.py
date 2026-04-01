"""
Bayesian Carry Strategy with Kelly Criterion Position Sizing

Trading strategy for short-dated prediction markets using:
- Bayesian posterior inference via NUTS (PyMC)
- Kelly criterion position sizing from posterior uncertainty
- Three hedging variants: baseline, volume-hedged, cash-hedged

Strategy Logic:
- Trade short-dated (≤30 days) No contracts
- BUY when posterior mean edge > min_edge
- Position size via fractional Kelly: f* = kelly_fraction * (μ_edge / σ²_edge)
- Wide posterior (high uncertainty) → small position
- Narrow posterior (high confidence) → large position

Hedging Variants:
1. Baseline (hedging=None): Pure Kelly-sized carry
2. Volume (hedging='volume'): Exit when volume z-score > threshold (informed trading signal)
3. Cash (hedging='cash'): Dynamic cash reserve that increases during drawdowns
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from collections import deque
from dataclasses import dataclass

from backtest.strategy import Strategy, Signal, SignalType
from models.bayesian_gamma_model.model import BayesianGammaModel
from models.bayesian_gamma_model.kelly_sizing import (
    get_posterior_cdf_samples,
    edge_distribution,
    kelly_size
)
from models.bayesian_gamma_model.bayesian_gamma_fitter import BayesianFitResult
from models.factored_gamma_model.empirical_bayes import EmpiricalBayesFactors
from models.factored_gamma_model.gamma_cdf_fitter import GammaCDFFitter
from backtest.data_loader import DataLoader


class BayesianCarryStrategy(Strategy):
    """
    Bayesian carry strategy with Kelly criterion position sizing.

    Trades short-dated (≤ 30 days) No contracts where Bayesian posterior
    indicates positive edge, sized via fractional Kelly criterion.

    Configuration Parameters:
        Model:
        - mcmc_draws: MCMC samples per chain (default: 1000)
        - mcmc_tune: MCMC tuning steps (default: 500)
        - mcmc_chains: Number of MCMC chains (default: 2)
        - mcmc_cores: CPU cores for MCMC (default: 4)
        - refit_days: Days between model refits (default: 7)

        Carry:
        - max_tte_days: Maximum days to expiration (default: 30)
        - min_edge: Minimum edge to take position (default: 0.05 = 5%)
        - kelly_fraction: Kelly fraction (default: 0.25 = quarter Kelly)
        - max_position: Max position as fraction of portfolio (default: 0.10 = 10%)

        Hedging:
        - hedging: None | 'volume' | 'cash' (default: None)

        Volume Hedge:
        - vol_lookback_hours: Hours of volume history (default: 24)
        - vol_accel_threshold: Z-score threshold for exit (default: 2.0)

        Cash Hedge:
        - min_cash_reserve: Base cash reserve (default: 0.20 = 20%)
        - drawdown_threshold: DD that triggers reserve increase (default: 0.05 = 5%)
        - drawdown_cash_add: Additional reserve during DD (default: 0.10 = 10%)
        - max_cash_reserve: Maximum cash reserve (default: 0.50 = 50%)
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize Bayesian carry strategy."""
        super().__init__(config)

        # Model parameters
        self.mcmc_draws = self.config.get('mcmc_draws', 1000)
        self.mcmc_tune = self.config.get('mcmc_tune', 500)
        self.mcmc_chains = self.config.get('mcmc_chains', 2)
        self.mcmc_cores = self.config.get('mcmc_cores', 4)
        self.refit_days = self.config.get('refit_days', 7)

        # Carry strategy parameters
        self.max_tte_days = self.config.get('max_tte_days', 30)
        self.min_edge = self.config.get('min_edge', 0.05)
        self.kelly_fraction = self.config.get('kelly_fraction', 0.25)
        self.max_position = self.config.get('max_position', 0.10)

        # Hedging variant
        self.hedging = self.config.get('hedging', None)  # None | 'volume' | 'cash'

        # Volume hedge parameters
        if self.hedging == 'volume':
            self.vol_lookback_hours = self.config.get('vol_lookback_hours', 24)
            self.vol_accel_threshold = self.config.get('vol_accel_threshold', 2.0)
            self.vol_history = {}  # market_id -> deque of volumes

        # Cash hedge parameters
        if self.hedging == 'cash':
            self.min_cash_reserve = self.config.get('min_cash_reserve', 0.20)
            self.drawdown_threshold = self.config.get('drawdown_threshold', 0.05)
            self.drawdown_cash_add = self.config.get('drawdown_cash_add', 0.10)
            self.max_cash_reserve = self.config.get('max_cash_reserve', 0.50)
            self.peak_value = self.config.get('initial_capital', 10000.0)

        # Initialize Bayesian Gamma model
        self.model = BayesianGammaModel(
            min_buckets=3,
            ci_level=0.70,  # Not used for Kelly, but needed for model
            mcmc_draws=self.mcmc_draws,
            mcmc_tune=self.mcmc_tune,
            mcmc_chains=self.mcmc_chains,
            mcmc_cores=self.mcmc_cores
        )

        # Empirical Bayes factors (for informative priors)
        self.use_eb_priors = self.config.get('use_eb_priors', True)
        self.eb_factors = None
        self.eb_fitter = GammaCDFFitter()  # For EB factor estimation

        if self.use_eb_priors:
            self.eb_holdout_end = self.config.get('eb_holdout_end_date', '2023-12-31')
            self.db_path = self.config.get('db_path', 'data/polymarket.db')
            # EB factors will be fit on first call if needed
            self._fit_eb_factors_lazy = True
        else:
            self._fit_eb_factors_lazy = False

        # State tracking
        self.fitted_events = {}  # event_id -> FitResult
        self.last_fit_date = {}  # event_id -> date
        self.verbose = self.config.get('verbose', False)

    @property
    def name(self) -> str:
        """Strategy name."""
        hedge_suffix = f"_{self.hedging}" if self.hedging else ""
        return f"BayesianCarry_TTE{self.max_tte_days}_Kelly{int(self.kelly_fraction*100)}{hedge_suffix}"

    def _fit_eb_factors(self):
        """
        Fit Empirical Bayes factors from historical resolved events.

        This learns category-level priors and feature adjustments from
        past events to inform priors for new events.
        """
        if not self.use_eb_priors:
            print("⚠ EB priors disabled in config")
            return

        print(f"\n{'='*70}")
        print("FITTING EMPIRICAL BAYES FACTORS")
        print(f"{'='*70}")
        print(f"Learning from historical events resolved before {self.eb_holdout_end}")
        print("This will create informative priors for:")
        print("  - Category-specific parameters (politics, crypto, sports)")
        print("  - Term structure feature adjustments (slope, curvature, rate)")
        print()

        # Load historical data
        from backtest.data_loader import DataLoader
        loader = DataLoader(self.db_path)

        # Get resolved events before holdout date
        try:
            historical_data = loader.load_timing_markets()
            print(f"Total data loaded: {len(historical_data)} rows")

            # Filter to resolved events before cutoff
            historical_data = historical_data[
                (historical_data['resolution_date'] <= self.eb_holdout_end) &
                (historical_data['resolution_date'].notna())
            ]

            print(f"After filtering (resolved before {self.eb_holdout_end}): {len(historical_data)} observations")
            print(f"Unique events: {historical_data['event_id'].nunique()}")

            if len(historical_data) == 0:
                print("✗ No historical data available for EB fitting")
                print("  Falling back to weak priors")
                self.use_eb_priors = False
                self.eb_factors = None
                return

            # Fit EB factors
            self.eb_factors = EmpiricalBayesFactors()
            self.eb_factors.fit(historical_data, self.eb_fitter, min_events_per_category=3)

            if self.eb_factors.fitted:
                print(f"\n✓ EB factors fitted successfully")
                print(f"  Categories: {list(self.eb_factors.categories)}")
                print(f"  Factors: {list(self.eb_factors.factors.keys())}")
            else:
                print(f"✗ EB fitting failed - no factors learned")
                self.use_eb_priors = False
                self.eb_factors = None

            print(f"{'='*70}\n")

        except Exception as e:
            import traceback
            print(f"✗ Failed to fit EB factors: {e}")
            print(f"  Traceback: {traceback.format_exc()}")
            print("  Falling back to weak priors")
            self.use_eb_priors = False
            self.eb_factors = None

    def _get_eb_priors(self, event_data: pd.DataFrame) -> dict:
        """
        Get EB-informed prior means for alpha and beta.

        Args:
            event_data: DataFrame for the event

        Returns:
            dict with 'alpha_prior_mean', 'beta_prior_mean', or None if EB not available
        """
        if not self.use_eb_priors or self.eb_factors is None or not self.eb_factors.fitted:
            return None

        # Extract category and features from event
        row = event_data.iloc[0]
        category = row.get('category', 'unknown')

        # Calculate term structure features (simplified - would need proper calc)
        # For now, use defaults or extract if available
        ts_slope = row.get('ts_slope', 0.0)
        ts_curvature = row.get('ts_curvature', 0.0)
        implied_rate = row.get('implied_rate', 0.1)

        # Get adjusted parameters from EB factors
        try:
            from models.factored_gamma_model.factor_adjustment import FactorAdjustment
            adjuster = FactorAdjustment(self.eb_factors)

            # Use base parameters and adjust
            alpha_base, beta_base = 2.0, 1.0  # Default base
            alpha_adj, beta_adj = adjuster.adjust(
                alpha_base, beta_base,
                category, ts_slope, ts_curvature, implied_rate
            )

            return {
                'alpha_prior_mean': alpha_adj,
                'beta_prior_mean': beta_adj,
                'category': category,
                'ts_slope': ts_slope,
                'ts_curvature': ts_curvature,
                'implied_rate': implied_rate
            }
        except Exception as e:
            if self.verbose:
                print(f"    ⚠ Failed to get EB priors: {e}")
            return None

    def _fit_event_with_eb(self, event_data: pd.DataFrame, current_date: pd.Timestamp,
                           event_id: str, eb_priors: dict = None):
        """
        Fit Bayesian model to event with EB-informed priors.

        This is a custom fit that passes EB priors to the fitter.
        """
        # Extract term structure (same as BayesianGammaModel)
        term_structure = self.model._extract_term_structure(event_data, current_date)

        if term_structure is None:
            return None

        times = term_structure['times']
        cdf_values = term_structure['cdf_values']
        market_ids = term_structure['market_ids']

        if len(times) < 3:
            return None

        # Check for previous posterior (sequential updating)
        prior_result = self.model.posterior_store.load_latest_posterior(event_id)

        if prior_result is not None:
            # Sequential fit with previous posterior as prior
            prior_date, prior_idata = prior_result
            bayesian_result = self.model.fitter.fit_sequential(times, cdf_values, prior_idata)
            is_sequential = True
        else:
            # Initial fit with EB-informed priors if available
            if eb_priors:
                bayesian_result = self.model.fitter.fit_initial(
                    times, cdf_values,
                    alpha_prior_mean=eb_priors['alpha_prior_mean'],
                    beta_prior_mean=eb_priors['beta_prior_mean'],
                    alpha_prior_std=3.0,
                    beta_prior_std=2.0
                )
            else:
                bayesian_result = self.model.fitter.fit_initial(times, cdf_values)
            is_sequential = False

        if bayesian_result is None:
            return None

        # Save posterior for sequential updating
        try:
            self.model.posterior_store.save_posterior(event_id, current_date, bayesian_result.idata)
        except Exception as e:
            if self.verbose:
                print(f"    ⚠ Failed to save posterior: {e}")

        # Compute credible intervals (needed for FitResult structure)
        posterior_samples = {
            'alpha': bayesian_result.idata.posterior['alpha'].values.flatten(),
            'beta': bayesian_result.idata.posterior['beta'].values.flatten()
        }

        ci_result = self.model.fitter.predict_cdf(times, posterior_samples, 0.70)

        # Map CI to market_ids
        credible_intervals = {}
        for i, (market_id, time) in enumerate(zip(market_ids, times)):
            credible_intervals[market_id] = {
                'lower': ci_result['lower'][i],
                'upper': ci_result['upper'][i],
                'median': ci_result['median'][i],
                'time': time
            }

        # Create FitResult (compatible with BayesianGammaModel)
        from models.bayesian_gamma_model.model import FitResult
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

    def _needs_refit(self, event_id: str, current_date: pd.Timestamp) -> bool:
        """Check if event needs refitting."""
        if event_id not in self.last_fit_date:
            return True
        days_since_fit = (current_date - self.last_fit_date[event_id]).days
        return days_since_fit >= self.refit_days

    def _hedge_triggered(self, row: pd.Series, current_date: pd.Timestamp) -> bool:
        """Check if hedge condition is triggered (variant-specific)."""
        if self.hedging == 'volume':
            return self._volume_hedge_check(row)
        return False

    def _volume_hedge_check(self, row: pd.Series) -> bool:
        """
        Check volume acceleration hedge.

        Exit/avoid positions when volume z-score > threshold.
        High volume acceleration may signal informed trading.
        """
        market_id = row['market_id']
        current_volume = row.get('volume_num', 0)

        # Initialize history if needed
        if market_id not in self.vol_history:
            self.vol_history[market_id] = deque(maxlen=self.vol_lookback_hours)

        history = self.vol_history[market_id]

        # Need at least 6 observations for meaningful z-score
        if len(history) < 6:
            history.append(current_volume)
            return False

        # Calculate z-score
        mean_vol = np.mean(history)
        std_vol = np.std(history)

        if std_vol < 1e-8:
            history.append(current_volume)
            return False

        z_score = (current_volume - mean_vol) / std_vol

        # Update history
        history.append(current_volume)

        return z_score > self.vol_accel_threshold

    def _capital_available(self) -> bool:
        """
        Check if capital is available (cash hedge variant).

        Returns True to allow position entry. The actual cash constraint
        is enforced by Portfolio class.
        """
        if self.hedging != 'cash':
            return True

        # Update peak value
        if self.portfolio_value > self.peak_value:
            self.peak_value = self.portfolio_value

        # Calculate drawdown
        drawdown = (self.peak_value - self.portfolio_value) / self.peak_value if self.peak_value > 0 else 0

        # Determine cash reserve (dynamic based on drawdown)
        if drawdown > self.drawdown_threshold:
            reserve = min(
                self.min_cash_reserve + self.drawdown_cash_add,
                self.max_cash_reserve
            )
        else:
            reserve = self.min_cash_reserve

        # Portfolio enforces hard constraints, we just signal availability
        # This is informational - actual enforcement happens in Portfolio.open_position()
        return True

    def on_data(self, current_date: pd.Timestamp, data: pd.DataFrame) -> List[Signal]:
        """
        Generate Kelly-sized carry trade signals.

        Process:
        1. Filter to short-dated markets (tte ≤ max_tte_days)
        2. Group by event, fit/refit Bayesian model
        3. Extract posterior CDF samples from storage
        4. Calculate Kelly size for each market
        5. Apply hedge filters
        6. Generate signals
        """
        signals = []

        # Fit EB factors on first call
        if self._fit_eb_factors_lazy:
            self._fit_eb_factors()
            self._fit_eb_factors_lazy = False

        # Get today's data
        today_data = data[data['date'] == current_date].copy()

        if today_data.empty:
            return signals

        # Calculate TTE for all markets (NO tte_days column exists!)
        today_data['tte_days'] = (today_data['resolution_date'] - current_date).dt.days

        # Filter to short-dated markets
        short_dated = today_data[today_data['tte_days'] <= self.max_tte_days]
        short_dated = short_dated[short_dated['tte_days'] > 0]  # Must have positive TTE

        if short_dated.empty:
            return signals

        # Group by event (use group_col if exists, else event_id)
        group_col = 'group_col' if 'group_col' in short_dated.columns else 'event_id'

        num_events = len(short_dated[group_col].unique())
        if self.verbose and num_events > 0:
            print(f"  Processing {num_events} events with {len(short_dated)} markets (tte ≤ {self.max_tte_days}d)")

        for event_id in short_dated[group_col].unique():
            event_data = short_dated[short_dated[group_col] == event_id]

            # Need at least 3 markets for term structure (min_buckets=3)
            num_markets = len(event_data['market_id'].unique())
            if num_markets < 3:
                if self.verbose and num_markets > 0:
                    print(f"  Skipping {event_id}: only {num_markets} markets (need ≥3)")
                continue

            # Check if refit needed
            if self._needs_refit(event_id, current_date):
                # Get EB-informed priors if available
                eb_priors = self._get_eb_priors(event_data)

                if self.verbose:
                    if eb_priors:
                        print(f"  Fitting {event_id} with EB-informed priors:")
                        print(f"    Category: {eb_priors.get('category', 'unknown')}")
                        print(f"    Prior α: {eb_priors['alpha_prior_mean']:.3f}, β: {eb_priors['beta_prior_mean']:.3f}")
                    else:
                        print(f"  Fitting {event_id} with weak priors (no EB)")
                        if self.use_eb_priors:
                            print(f"    (EB enabled but priors unavailable for this event)")

                # Fit event with EB-informed priors
                try:
                    fit_result = self._fit_event_with_eb(event_data, current_date, event_id, eb_priors)
                except Exception as e:
                    if self.verbose:
                        print(f"    ✗ Fit failed for event_id={event_id}")
                        print(f"      Error type: {type(e).__name__}")
                        print(f"      Error: {str(e)[:200]}")
                        print(f"      Markets: {len(event_data['market_id'].unique())}")
                        print(f"      TTE range: {event_data['tte_days'].min():.1f} - {event_data['tte_days'].max():.1f} days")
                    continue

                if fit_result is None:
                    if self.verbose:
                        print(f"    ✗ Fit returned None for event_id={event_id}")
                        print(f"      Markets: {len(event_data['market_id'].unique())}")
                        print(f"      Price range: {event_data['price'].min():.3f} - {event_data['price'].max():.3f}")
                        print(f"      TTE range: {event_data['tte_days'].min():.1f} - {event_data['tte_days'].max():.1f} days")
                    continue

                self.fitted_events[event_id] = fit_result
                self.last_fit_date[event_id] = current_date

                if self.verbose:
                    conv_status = "✓" if fit_result.converged else "⚠"
                    print(f"    {conv_status} Fit: α={fit_result.alpha_mean:.3f}±{fit_result.alpha_std:.3f}, "
                          f"β={fit_result.beta_mean:.3f}±{fit_result.beta_std:.3f}, "
                          f"converged={fit_result.converged}")

            # Get fit result
            if event_id not in self.fitted_events:
                continue

            fit_result = self.fitted_events[event_id]

            # Load posterior from storage for Kelly sizing
            # Model saves idata to PosteriorStore in fit_event()
            posterior_result = self.model.posterior_store.load_latest_posterior(event_id)

            if posterior_result is None:
                continue

            posterior_date, idata = posterior_result

            # Create BayesianFitResult-like object for kelly_sizing functions
            @dataclass
            class TempBayesianResult:
                idata: any

            temp_result = TempBayesianResult(idata=idata)

            # Get posterior CDF samples for Kelly sizing
            try:
                posterior_cdf_samples = get_posterior_cdf_samples(
                    temp_result,
                    fit_result.times,
                    n_samples=1000
                )
            except Exception as e:
                if self.verbose:
                    print(f"    ✗ Failed to get posterior samples for {event_id}: {e}")
                continue

            # Get market prices for this event (No prices)
            event_no_data = event_data[event_data['outcome'] == 'No'].copy()
            event_no_data = event_no_data.sort_values('tte_days')

            if event_no_data.empty:
                continue

            market_prices = event_no_data['price'].values

            # PASS 1: Collect potential signals
            potential_signals = []

            for tenor_idx, (_, row) in enumerate(event_no_data.iterrows()):
                market_id = row['market_id']

                # Hedge check (exit positions if triggered)
                if self._hedge_triggered(row, current_date):
                    # Close existing position if any
                    current_position = self.get_position(market_id, 'No')
                    if current_position > 0:
                        signals.append(Signal(
                            market_id=market_id,
                            token_id=row['token_id'],
                            outcome='No',
                            signal_type=SignalType.SELL,
                            size=current_position,
                            price=row['price'],
                            reason=f"Volume hedge triggered (z>{self.vol_accel_threshold})",
                            metadata={'hedge_type': 'volume'}
                        ))
                    continue

                # Cash reserve check
                if not self._capital_available():
                    continue

                # Calculate Kelly size
                try:
                    size_dollars = kelly_size(
                        posterior_cdf_samples,
                        market_prices,
                        tenor_idx,
                        self.portfolio_value,
                        fraction=self.kelly_fraction,
                        max_position=self.max_position,
                        min_edge=self.min_edge
                    )
                except Exception as e:
                    if self.verbose:
                        print(f"    ✗ Kelly sizing failed for {market_id}: {e}")
                    continue

                if size_dollars <= 0:
                    continue

                # Check if we have position
                current_position = self.get_position(market_id, 'No')

                # Only enter new positions (no rebalancing)
                if current_position == 0:
                    mu_edge, var_edge, _ = edge_distribution(
                        posterior_cdf_samples, market_prices, tenor_idx
                    )

                    if self.verbose and mu_edge > 0.005:  # Log edges > 0.5%
                        print(f"    → {row['market_id'][:20]}... edge={mu_edge:.3f}, var={var_edge:.4f}, "
                              f"kelly=${size_dollars:.0f}, price={row['price']:.3f}, tte={row['tte_days']}d")

                    potential_signals.append({
                        'row': row,
                        'tenor_idx': tenor_idx,
                        'size': size_dollars,
                        'mu_edge': mu_edge,
                        'var_edge': var_edge
                    })

            # PASS 2: Generate signals
            for signal_info in potential_signals:
                row = signal_info['row']
                size = signal_info['size']
                mu_edge = signal_info['mu_edge']
                var_edge = signal_info['var_edge']

                signals.append(Signal(
                    market_id=row['market_id'],
                    token_id=row['token_id'],
                    outcome='No',
                    signal_type=SignalType.BUY,
                    size=size,
                    price=row['price'],
                    reason=f"Carry edge={mu_edge:.3f}, kelly=${size:.0f}, tte={row['tte_days']}d",
                    metadata={
                        'mu_edge': mu_edge,
                        'var_edge': var_edge,
                        'kelly_size': size,
                        'tte_days': row['tte_days'],
                        'alpha_mean': fit_result.alpha_mean,
                        'beta_mean': fit_result.beta_mean,
                        'hedging': self.hedging
                    }
                ))

        return signals

    def on_market_close(self, market_id: str, outcome: str, final_price: float):
        """Clean up volume history when market closes."""
        super().on_market_close(market_id, outcome, final_price)

        if self.hedging == 'volume' and market_id in self.vol_history:
            del self.vol_history[market_id]

    def reset(self):
        """Reset strategy state."""
        super().reset()
        self.fitted_events = {}
        self.last_fit_date = {}

        if self.hedging == 'volume':
            self.vol_history = {}

        if self.hedging == 'cash':
            self.peak_value = self.config.get('initial_capital', 10000.0)
