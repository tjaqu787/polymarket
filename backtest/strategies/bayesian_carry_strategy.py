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

        # State tracking
        self.fitted_events = {}  # event_id -> FitResult
        self.last_fit_date = {}  # event_id -> date
        self.verbose = self.config.get('verbose', False)

    @property
    def name(self) -> str:
        """Strategy name."""
        hedge_suffix = f"_{self.hedging}" if self.hedging else ""
        return f"BayesianCarry_TTE{self.max_tte_days}_Kelly{int(self.kelly_fraction*100)}{hedge_suffix}"

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

        for event_id in short_dated[group_col].unique():
            event_data = short_dated[short_dated[group_col] == event_id]

            # Need at least 2 markets for term structure
            if len(event_data['market_id'].unique()) < 2:
                continue

            # Check if refit needed
            if self._needs_refit(event_id, current_date):
                if self.verbose:
                    print(f"  Fitting {event_id} with NUTS (Kelly carry strategy)...")

                # Fit event (returns FitResult) - with error handling
                try:
                    fit_result = self.model.fit_event(event_data, current_date, event_id)
                except Exception as e:
                    if self.verbose:
                        print(f"    ✗ Fit failed: {e}")
                    continue

                if fit_result is None:
                    if self.verbose:
                        print(f"    ✗ Fit returned None (insufficient data or numerical issues)")
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
