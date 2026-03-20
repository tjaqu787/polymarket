"""
Poisson Timing Strategy for time-distributed events.

Uses the Poisson Timing Model to predict event timing and trade accordingly.
"""

import sys
sys.path.append('.')

import pandas as pd
import numpy as np
from typing import List, Dict, Optional
from datetime import datetime, timedelta
import re

from backtest.strategy import Strategy, Signal, SignalType
from models.poisson_timing_model import PoissonTimingModel


class PoissonTimingStrategy(Strategy):
    """
    Calendar Call/Put Strategy using Poisson timing model credible intervals.

    Strategy logic (like a calendar spread):
    1. Identify time-distributed events (markets with "by <date>" in question)
    2. Fit Poisson model to current price term structure with credible intervals
    3. BUY when market price < lower bound of 70% CI (like buying calls - underpriced)
    4. SHORT when market price > upper bound of 70% CI (like selling puts - overpriced)
    5. SELL (close longs) when price reaches upper bound
    6. COVER (close shorts) when price reaches lower bound
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)

        # Strategy parameters
        self.ci_level = self.config.get('ci_level', 0.70)  # Credible interval level (70%)
        self.min_buckets = self.config.get('min_buckets', 3)  # Min time buckets needed
        self.max_rmse = self.config.get('max_rmse', 0.3)  # Max fitting error
        self.distribution = self.config.get('distribution', 'gamma')  # Distribution type
        self.n_bootstrap = self.config.get('n_bootstrap', 500)  # Bootstrap samples for CI
        self.max_event_exposure = self.config.get('max_event_exposure', 0.15)  # Max % of portfolio per event

        # Track events we've analyzed
        self.fitted_events = {}  # event_id -> last fit result
        self.last_fit_date = {}  # event_id -> date of last fit
        self.refit_days = self.config.get('refit_days', 7)  # Refit every N days

        self.model = PoissonTimingModel(distribution=self.distribution)

    @property
    def name(self) -> str:
        return f"PoissonTiming_{self.distribution}_CI{int(self.ci_level*100)}"

    def extract_target_date(self, question: str) -> Optional[datetime]:
        """Extract target date from question text."""
        patterns = [
            r'by ([A-Za-z]+) (\d+), (\d{4})',  # "by March 15, 2026"
            r'before ([A-Za-z]+) (\d+), (\d{4})',  # "before March 15, 2026"
        ]
        for pattern in patterns:
            match = re.search(pattern, question)
            if match:
                try:
                    month_name, day, year = match.groups()
                    date_str = f"{year}-{month_name}-{day}"
                    return datetime.strptime(date_str, '%Y-%B-%d')
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

    def fit_event_timing(
        self,
        event_data: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> Optional[Dict]:
        """
        Fit Poisson timing model to an event.

        Returns:
            Dictionary with fit results or None if fitting failed
        """
        # Extract target dates and prices
        markets = []
        for _, row in event_data.iterrows():
            if 'by' not in row['question'].lower():
                continue

            target = self.extract_target_date(row['question'])
            if target is None or row['outcome'] != 'Yes':
                continue

            # Calculate time to target
            days_to_target = (target - current_date.to_pydatetime()).days
            if days_to_target <= 0:  # Skip past targets
                continue

            markets.append({
                'market_id': row['market_id'],
                'target_date': target,
                'days_to_target': days_to_target,
                'years_to_target': days_to_target / 365.25,
                'price': row['price']
            })

        if len(markets) < self.min_buckets:
            return None

        # Sort by time and remove duplicates
        df = pd.DataFrame(markets).drop_duplicates(subset=['years_to_target'])
        df = df.sort_values('years_to_target')

        if len(df) < self.min_buckets:
            return None

        times = df['years_to_target'].values
        cdf_values = df['price'].values

        # Validate data
        if np.any(np.isnan(times)) or np.any(np.isnan(cdf_values)):
            return None
        if np.any(np.diff(times) == 0):
            return None

        try:
            # Fit model
            fit_result = self.model.fit_mle(times, cdf_values)

            # Check goodness of fit
            if fit_result['rmse'] > self.max_rmse:
                return None

            # Calculate credible intervals
            credible_intervals = self.model.calculate_credible_intervals(
                times,
                cdf_values,
                times,  # Evaluate at same time points
                ci_level=self.ci_level,
                n_bootstrap=self.n_bootstrap
            )

            return {
                'params': fit_result['params'],
                'rmse': fit_result['rmse'],
                'aic': fit_result['aic'],
                'credible_intervals': credible_intervals,
                'markets': df,
                'fit_date': current_date,
                'times': times,
                'cdf_values': cdf_values
            }

        except Exception as e:
            # Fitting failed
            return None

    def calculate_credible_bounds(
        self,
        market_id: str,
        fit_result: Dict,
        current_date: pd.Timestamp
    ) -> Optional[Dict]:
        """
        Calculate credible interval bounds for a market based on fitted model.

        Returns:
            Dictionary with 'lower', 'upper', 'median' or None if not found
        """
        # Find this market in the fitted markets
        markets_df = fit_result['markets']
        market_row = markets_df[markets_df['market_id'] == market_id]

        if len(market_row) == 0:
            return None

        target_time_years = market_row.iloc[0]['years_to_target']

        # Find the index in the fit result times that's closest to this market's time
        times = fit_result['times']
        idx = np.argmin(np.abs(times - target_time_years))

        ci = fit_result['credible_intervals']

        return {
            'lower': ci['lower'][idx],
            'upper': ci['upper'][idx],
            'median': ci['median'][idx]
        }

    def _calculate_position_size(self, price: float, exposure_fraction: float) -> float:
        """
        Calculate position size (number of contracts) based on exposure fraction.

        Args:
            price: Entry price per contract
            exposure_fraction: Fraction of portfolio to allocate (e.g., 0.03 for 3%)

        Returns:
            Number of contracts to buy
        """
        if price <= 0 or self.portfolio_value <= 0:
            return 0.0

        # Calculate target allocation in dollars
        target_allocation = self.portfolio_value * exposure_fraction

        # Calculate number of contracts
        size = target_allocation / price

        # Round down to avoid exceeding allocation
        size = int(size)

        return max(0.0, float(size))

    def _calculate_event_position_sizes(
        self,
        potential_signals: List[Dict],
        current_date: pd.Timestamp
    ) -> List[Signal]:
        """
        Calculate position sizes for signals based on event-level exposure limits.

        Args:
            potential_signals: List of signal info dicts from first pass
            current_date: Current date

        Returns:
            List of Signal objects with calculated sizes
        """
        signals = []

        # Count only new position signals (BUY/SHORT) for sizing
        new_position_signals = [
            s for s in potential_signals
            if s['type'] in [SignalType.BUY, SignalType.SHORT]
        ]

        num_new_positions = len(new_position_signals)

        # Calculate per-market exposure
        if num_new_positions > 0:
            per_market_exposure = self.max_event_exposure / num_new_positions
        else:
            per_market_exposure = 0.0

        # Generate signals with calculated sizes
        for signal_info in potential_signals:
            row = signal_info['row']
            ci_bounds = signal_info['ci_bounds']
            fit_result = signal_info['fit_result']
            signal_type = signal_info['type']

            # Calculate size based on signal type
            if signal_type in [SignalType.BUY, SignalType.SHORT]:
                # NEW POSITION: Calculate size based on event exposure
                size = self._calculate_position_size(
                    price=row['price'],
                    exposure_fraction=per_market_exposure
                )

                reason_prefix = "Below" if signal_type == SignalType.BUY else "Above"
                reason = (
                    f"{reason_prefix} {int(self.ci_level*100)}% CI: "
                    f"Price={row['price']:.3f}, "
                    f"Lower={ci_bounds['lower']:.3f}, "
                    f"Upper={ci_bounds['upper']:.3f}, "
                    f"Allocation={per_market_exposure*100:.1f}%"
                )
            else:
                # EXIT POSITION (SELL/COVER): Use existing position size
                size = signal_info['position']

                reason_prefix = "Exit long" if signal_type == SignalType.SELL else "Exit short"
                bound_type = "upper" if signal_type == SignalType.SELL else "lower"
                bound_value = ci_bounds['upper'] if signal_type == SignalType.SELL else ci_bounds['lower']
                reason = (
                    f"{reason_prefix} at {bound_type} bound: "
                    f"Price={row['price']:.3f}, "
                    f"{bound_type.capitalize()}={bound_value:.3f}"
                )

            signals.append(Signal(
                market_id=row['market_id'],
                token_id=row['token_id'],
                outcome='Yes',
                signal_type=signal_type,
                size=size,
                price=row['price'],
                reason=reason,
                metadata={
                    'lower_bound': ci_bounds['lower'],
                    'upper_bound': ci_bounds['upper'],
                    'median': ci_bounds['median'],
                    'rmse': fit_result['rmse'],
                    'event_exposure': per_market_exposure if signal_type in [SignalType.BUY, SignalType.SHORT] else None,
                    'num_event_positions': num_new_positions if signal_type in [SignalType.BUY, SignalType.SHORT] else None
                }
            ))

        return signals

    def on_data(self, current_date: pd.Timestamp, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals based on Poisson timing model."""
        signals = []

        # Get data for current date
        today_data = data[data['date'] == current_date]

        if today_data.empty:
            return signals

        # Group by event
        for event_id in today_data['event_id'].unique():
            event_data = today_data[today_data['event_id'] == event_id]

            # Check if this is a time-distributed event
            if not self.is_time_distributed_event(event_data):
                continue

            # Check if we need to (re)fit the model
            need_fit = (
                event_id not in self.fitted_events or
                event_id not in self.last_fit_date or
                (current_date - self.last_fit_date[event_id]).days >= self.refit_days
            )

            if need_fit:
                fit_result = self.fit_event_timing(event_data, current_date)
                if fit_result is not None:
                    self.fitted_events[event_id] = fit_result
                    self.last_fit_date[event_id] = current_date
                else:
                    # Fitting failed, skip this event
                    continue

            # Get fitted model for this event
            if event_id not in self.fitted_events:
                continue

            fit_result = self.fitted_events[event_id]

            # PASS 1: Collect all markets that would trigger signals
            potential_signals = []
            for _, row in event_data[event_data['outcome'] == 'Yes'].iterrows():
                market_id = row['market_id']
                market_price = row['price']

                # Calculate credible interval bounds
                ci_bounds = self.calculate_credible_bounds(market_id, fit_result, current_date)

                if ci_bounds is None:
                    continue

                lower_bound = ci_bounds['lower']
                upper_bound = ci_bounds['upper']
                current_position = self.get_position(market_id, 'Yes')

                # Check which signal type this would trigger
                signal_info = None

                if market_price < lower_bound and current_position == 0:
                    signal_info = {
                        'type': SignalType.BUY,
                        'row': row,
                        'ci_bounds': ci_bounds,
                        'fit_result': fit_result
                    }
                elif market_price > upper_bound and current_position == 0:
                    signal_info = {
                        'type': SignalType.SHORT,
                        'row': row,
                        'ci_bounds': ci_bounds,
                        'fit_result': fit_result
                    }
                elif market_price >= upper_bound and current_position > 0:
                    signal_info = {
                        'type': SignalType.SELL,
                        'row': row,
                        'ci_bounds': ci_bounds,
                        'fit_result': fit_result,
                        'position': current_position
                    }
                elif market_price <= lower_bound and current_position < 0:
                    signal_info = {
                        'type': SignalType.COVER,
                        'row': row,
                        'ci_bounds': ci_bounds,
                        'fit_result': fit_result,
                        'position': abs(current_position)
                    }

                if signal_info is not None:
                    potential_signals.append(signal_info)

            # PASS 2: Calculate sizes and generate signals
            event_signals = self._calculate_event_position_sizes(
                potential_signals,
                current_date
            )

            signals.extend(event_signals)

        return signals

    def reset(self):
        """Reset strategy state."""
        super().reset()
        self.fitted_events = {}
        self.last_fit_date = {}
