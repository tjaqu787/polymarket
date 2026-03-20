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

            # Generate signals for each market in this event
            for _, row in event_data[event_data['outcome'] == 'Yes'].iterrows():
                market_id = row['market_id']
                market_price = row['price']

                # Calculate credible interval bounds
                ci_bounds = self.calculate_credible_bounds(market_id, fit_result, current_date)

                if ci_bounds is None:
                    continue

                lower_bound = ci_bounds['lower']
                upper_bound = ci_bounds['upper']
                median = ci_bounds['median']

                current_position = self.get_position(market_id, 'Yes')

                # BUY signal: price below lower bound (like buying a call - underpriced)
                if market_price < lower_bound and current_position == 0:
                    signals.append(Signal(
                        market_id=market_id,
                        token_id=row['token_id'],
                        outcome='Yes',
                        signal_type=SignalType.BUY,
                        size=1.0,
                        price=market_price,
                        reason=f"Below {int(self.ci_level*100)}% CI: Price={market_price:.3f}, Lower={lower_bound:.3f}, Upper={upper_bound:.3f}",
                        metadata={
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound,
                            'median': median,
                            'rmse': fit_result['rmse']
                        }
                    ))

                # SHORT signal: price above upper bound (like selling a put - overpriced)
                elif market_price > upper_bound and current_position == 0:
                    signals.append(Signal(
                        market_id=market_id,
                        token_id=row['token_id'],
                        outcome='Yes',
                        signal_type=SignalType.SHORT,
                        size=1.0,
                        price=market_price,
                        reason=f"Above {int(self.ci_level*100)}% CI: Price={market_price:.3f}, Lower={lower_bound:.3f}, Upper={upper_bound:.3f}",
                        metadata={
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound,
                            'median': median,
                            'rmse': fit_result['rmse']
                        }
                    ))

                # SELL signal: close long position when price reaches upper bound
                elif market_price >= upper_bound and current_position > 0:
                    signals.append(Signal(
                        market_id=market_id,
                        token_id=row['token_id'],
                        outcome='Yes',
                        signal_type=SignalType.SELL,
                        size=current_position,
                        price=market_price,
                        reason=f"Exit long at upper bound: Price={market_price:.3f}, Upper={upper_bound:.3f}",
                        metadata={
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound,
                            'median': median,
                            'rmse': fit_result['rmse']
                        }
                    ))

                # COVER signal: close short position when price reaches lower bound
                elif market_price <= lower_bound and current_position < 0:
                    signals.append(Signal(
                        market_id=market_id,
                        token_id=row['token_id'],
                        outcome='Yes',
                        signal_type=SignalType.COVER,
                        size=abs(current_position),
                        price=market_price,
                        reason=f"Exit short at lower bound: Price={market_price:.3f}, Lower={lower_bound:.3f}",
                        metadata={
                            'lower_bound': lower_bound,
                            'upper_bound': upper_bound,
                            'median': median,
                            'rmse': fit_result['rmse']
                        }
                    ))

        return signals

    def reset(self):
        """Reset strategy state."""
        super().reset()
        self.fitted_events = {}
        self.last_fit_date = {}
