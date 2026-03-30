"""
Bayesian Gamma Timing Strategy

Trading strategy using the Bayesian Gamma Model with sequential updating.

Key Differences from FactoredGammaStrategy:
- Uses PyMC MCMC instead of MLE + bootstrap
- Sequential Bayesian updating: previous posterior → new prior
- True Bayesian credible intervals from posterior samples
- Multithreaded MCMC sampling for speed

Trading Logic (same as FactoredGamma):
- BUY when market_price < CI_lower (underpriced)
- SHORT when market_price > CI_upper (overpriced)
- SELL when long position reaches CI_upper
- COVER when short position reaches CI_lower

Position Sizing:
    size = max_event_exposure / (num_signals * interval_width)
    Narrow CI (confident) → larger size
    Wide CI (uncertain) → smaller size
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime
import re

from backtest.strategy import Strategy, Signal, SignalType
from backtest.data_loader import DataLoader
from models.bayesian_gamma_model.model import BayesianGammaModel


class BayesianGammaStrategy(Strategy):
    """
    Bayesian Gamma timing strategy with sequential updating.

    Uses PyMC MCMC for true Bayesian inference with posterior → prior chaining.

    Configuration Parameters:
        - db_path: Path to polymarket.db (default: 'data/polymarket.db')
        - min_buckets: Min term structure points (default: 3)
        - ci_level: Credible interval level (default: 0.70)
        - mcmc_draws: MCMC samples per chain (default: 500)
        - mcmc_tune: MCMC tuning steps (default: 500)
        - mcmc_chains: Number of MCMC chains (default: 2)
        - mcmc_cores: CPU cores for MCMC (default: 4, max 12)
        - refit_hours: Hours between refits (default: 6)
        - max_event_exposure: Max portfolio % per event (default: 0.15)
        - posterior_dir: Directory for storing posteriors
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Bayesian Gamma strategy."""
        super().__init__(config)

        # Extract config parameters
        self.min_buckets = self.config.get('min_buckets', 3)
        self.ci_level = self.config.get('ci_level', 0.70)
        self.mcmc_draws = self.config.get('mcmc_draws', 500)
        self.mcmc_tune = self.config.get('mcmc_tune', 500)
        self.mcmc_chains = self.config.get('mcmc_chains', 2)
        self.mcmc_cores = self.config.get('mcmc_cores', 4)  # Use 4 cores by default
        self.refit_hours = self.config.get('refit_hours', 6)
        self.max_event_exposure = self.config.get('max_event_exposure', 0.15)
        self.posterior_dir = self.config.get('posterior_dir', 'models/bayesian_gamma_model/posteriors')

        # Initialize model
        self.model = BayesianGammaModel(
            min_buckets=self.min_buckets,
            ci_level=self.ci_level,
            mcmc_draws=self.mcmc_draws,
            mcmc_tune=self.mcmc_tune,
            mcmc_chains=self.mcmc_chains,
            posterior_dir=self.posterior_dir
        )

        # Track fitted events
        self.fitted_events = {}  # event_id -> FitResult
        self.last_fit_date = {}  # event_id -> date

        # Data loader
        db_path = self.config.get('db_path', 'data/polymarket.db')
        self.data_loader = DataLoader(db_path)

    @property
    def name(self) -> str:
        """Return strategy name for logging."""
        return f"BayesianGamma_CI{int(self.ci_level*100)}"

    def extract_target_date(self, question: str) -> Optional[datetime]:
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

    def is_time_distributed_event(self, event_data: pd.DataFrame) -> bool:
        """Check if an event is time-distributed (has multiple "by X" markets)."""
        questions_with_by = event_data[
            event_data['question'].str.contains(' by ', case=False, na=False)
        ]
        unique_target_dates = set()
        for q in questions_with_by['question']:
            target = self.extract_target_date(q)
            if target:
                unique_target_dates.add(target)
        return len(unique_target_dates) >= self.min_buckets

    def _calculate_position_size(self, price: float, exposure_fraction: float) -> float:
        """
        Calculate position size (number of contracts) based on exposure fraction.

        Args:
            price: Entry price per contract
            exposure_fraction: Fraction of portfolio to allocate

        Returns:
            Number of contracts to buy
        """
        if price <= 0 or self.portfolio_value <= 0:
            return 0.0

        target_allocation = self.portfolio_value * exposure_fraction
        size = target_allocation / price
        size = int(size)

        return max(0.0, float(size))

    def _calculate_event_position_sizes(
        self,
        potential_signals: List[Dict],
        event_id: str
    ) -> List[Signal]:
        """
        Calculate position sizes based on interval width.

        Position sizing inversely proportional to interval width:
            size ∝ 1 / interval_width

        Narrow CI (confident) → larger size
        Wide CI (uncertain) → smaller size

        Args:
            potential_signals: List of signal dicts
            event_id: Event identifier

        Returns:
            List of Signal objects with calculated sizes
        """
        signals = []

        # Separate new positions from exits
        new_position_signals = [
            s for s in potential_signals
            if s['type'] in [SignalType.BUY, SignalType.SHORT]
        ]

        # Calculate position sizes for new positions
        if len(new_position_signals) > 0:
            # Calculate total inverse interval width (for normalization)
            total_inv_width = sum(
                1.0 / max(s['prediction'].interval_width, 0.01)
                for s in new_position_signals
            )

            # Allocate exposure proportional to confidence
            for signal_info in new_position_signals:
                row = signal_info['row']
                prediction = signal_info['prediction']
                signal_type = signal_info['type']

                # Weight by inverse interval width
                inv_width = 1.0 / max(prediction.interval_width, 0.01)
                weight = inv_width / total_inv_width

                # Allocate fraction of max_event_exposure
                per_market_exposure = self.max_event_exposure * weight

                # Calculate size
                size = self._calculate_position_size(
                    price=row['price'],
                    exposure_fraction=per_market_exposure
                )

                reason_prefix = "Below" if signal_type == SignalType.BUY else "Above"
                reason = (
                    f"{reason_prefix} {int(self.ci_level*100)}% CI (Bayesian): "
                    f"Price={row['price']:.3f}, "
                    f"Lower={prediction.lower_bound:.3f}, "
                    f"Upper={prediction.upper_bound:.3f}, "
                    f"Width={prediction.interval_width:.3f}, "
                    f"Allocation={per_market_exposure*100:.1f}%"
                )

                signals.append(Signal(
                    market_id=row['market_id'],
                    token_id=row['token_id'],
                    outcome='No',
                    signal_type=signal_type,
                    size=size,
                    price=row['price'],
                    reason=reason,
                    metadata={
                        'lower_bound': prediction.lower_bound,
                        'upper_bound': prediction.upper_bound,
                        'median': prediction.median,
                        'interval_width': prediction.interval_width,
                        'alpha_mean': prediction.alpha_mean,
                        'beta_mean': prediction.beta_mean,
                        'event_exposure': per_market_exposure,
                        'event_id': event_id,
                        'method': 'bayesian_mcmc'
                    }
                ))

        # Process exit signals (SELL/COVER)
        exit_signals = [
            s for s in potential_signals
            if s['type'] in [SignalType.SELL, SignalType.COVER]
        ]

        for signal_info in exit_signals:
            row = signal_info['row']
            prediction = signal_info['prediction']
            signal_type = signal_info['type']
            position_size = signal_info['position']

            reason_prefix = "Exit long" if signal_type == SignalType.SELL else "Exit short"
            bound_type = "upper" if signal_type == SignalType.SELL else "lower"
            bound_value = prediction.upper_bound if signal_type == SignalType.SELL else prediction.lower_bound

            reason = (
                f"{reason_prefix} at {bound_type} bound (Bayesian): "
                f"Price={row['price']:.3f}, "
                f"{bound_type.capitalize()}={bound_value:.3f}"
            )

            signals.append(Signal(
                market_id=row['market_id'],
                token_id=row['token_id'],
                outcome='No',
                signal_type=signal_type,
                size=position_size,
                price=row['price'],
                reason=reason,
                metadata={
                    'lower_bound': prediction.lower_bound,
                    'upper_bound': prediction.upper_bound,
                    'median': prediction.median,
                    'interval_width': prediction.interval_width,
                    'event_id': event_id,
                    'method': 'bayesian_mcmc'
                }
            ))

        return signals

    def on_data(self, current_date: pd.Timestamp, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on Bayesian Gamma model.

        Implementation:
            1. EVERY TIMESTEP:
                - Group data by event
                - Filter to time-distributed events
                - Every refit_hours: Call model.fit_event() (MCMC with sequential updating)
                - For each market: Get prediction, generate signals
            2. Calculate position sizes using interval_width

        Note: MCMC is parallelized across cores (set mcmc_cores in config)
        """
        signals = []

        # Get data for current date
        today_data = data[data['date'] == current_date]

        if today_data.empty:
            return signals

        # Group by semantic group (or fallback to event_id)
        group_col = 'group_col' if 'group_col' in today_data.columns else 'event_id'

        for group_id in today_data[group_col].unique():
            event_data = today_data[today_data[group_col] == group_id]

            # Check if this is a time-distributed event
            if not self.is_time_distributed_event(event_data):
                continue

            # Check if we need to (re)fit the model
            need_fit = (
                group_id not in self.fitted_events or
                group_id not in self.last_fit_date or
                (current_date - self.last_fit_date[group_id]).total_seconds() / 3600 >= self.refit_hours
            )

            if need_fit:
                # Fit event using Bayesian Gamma Model with sequential updating
                print(f"  Fitting {group_id} with Bayesian MCMC (using {self.mcmc_cores} cores)...")

                fit_result = self.model.fit_event(event_data, current_date, group_id)

                if fit_result is not None:
                    self.fitted_events[group_id] = fit_result
                    self.last_fit_date[group_id] = current_date

                    # Log convergence info
                    seq_status = "sequential" if fit_result.is_sequential else "initial"
                    conv_status = "✓" if fit_result.converged else "✗"
                    print(f"    {conv_status} {seq_status} fit: "
                          f"α={fit_result.alpha_mean:.3f}±{fit_result.alpha_std:.3f}, "
                          f"β={fit_result.beta_mean:.3f}±{fit_result.beta_std:.3f}, "
                          f"R̂_α={fit_result.rhat_alpha:.3f}, R̂_β={fit_result.rhat_beta:.3f}")
                else:
                    # Fitting failed, skip this event
                    continue

            # Get fitted model for this event
            if group_id not in self.fitted_events:
                continue

            fit_result = self.fitted_events[group_id]

            # PASS 1: Collect all markets that would trigger signals
            potential_signals = []

            for _, row in event_data[event_data['outcome'] == 'No'].iterrows():
                market_id = row['market_id']
                market_price = row['price']

                # Get prediction from model
                prediction = self.model.predict(market_id, fit_result, current_date)

                if prediction is None:
                    continue

                lower_bound = prediction.lower_bound
                upper_bound = prediction.upper_bound
                current_position = self.get_position(market_id, 'No')

                # Check which signal type this would trigger
                signal_info = None

                if market_price < lower_bound and current_position == 0:
                    # BUY signal: price below lower bound (underpriced)
                    signal_info = {
                        'type': SignalType.BUY,
                        'row': row,
                        'prediction': prediction
                    }
                elif market_price > upper_bound and current_position == 0:
                    # SHORT signal: price above upper bound (overpriced)
                    signal_info = {
                        'type': SignalType.SHORT,
                        'row': row,
                        'prediction': prediction
                    }
                elif market_price >= upper_bound and current_position > 0:
                    # SELL signal: exit long position at upper bound
                    signal_info = {
                        'type': SignalType.SELL,
                        'row': row,
                        'prediction': prediction,
                        'position': current_position
                    }
                elif market_price <= lower_bound and current_position < 0:
                    # COVER signal: exit short position at lower bound
                    signal_info = {
                        'type': SignalType.COVER,
                        'row': row,
                        'prediction': prediction,
                        'position': abs(current_position)
                    }

                if signal_info is not None:
                    potential_signals.append(signal_info)

            # PASS 2: Calculate sizes and generate signals
            event_signals = self._calculate_event_position_sizes(
                potential_signals,
                group_id
            )

            signals.extend(event_signals)

        return signals

    def reset(self):
        """Reset strategy state."""
        super().reset()
        self.fitted_events = {}
        self.last_fit_date = {}
