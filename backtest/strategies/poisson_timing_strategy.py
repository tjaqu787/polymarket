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
    Strategy that uses Poisson timing model to predict event timing
    and trades time-distributed markets accordingly.

    Strategy logic:
    1. Identify time-distributed events (markets with "by <date>" in question)
    2. Fit Poisson model to current price term structure
    3. Buy markets where predicted probability > market price (underpriced)
    4. Sell when market price > predicted probability (overpriced)
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)

        # Strategy parameters
        self.min_edge = self.config.get('min_edge', 0.05)  # Minimum edge to trade (5%)
        self.min_buckets = self.config.get('min_buckets', 3)  # Min time buckets needed
        self.max_rmse = self.config.get('max_rmse', 0.3)  # Max fitting error
        self.distribution = self.config.get('distribution', 'gamma')  # Distribution type

        # Track events we've analyzed
        self.fitted_events = {}  # event_id -> last fit result
        self.last_fit_date = {}  # event_id -> date of last fit
        self.refit_days = self.config.get('refit_days', 7)  # Refit every N days

        self.model = PoissonTimingModel(distribution=self.distribution)

    @property
    def name(self) -> str:
        return f"PoissonTiming_{self.distribution}_{self.min_edge}"

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

            # Make predictions
            pred_times = np.linspace(times.min(), times.max(), 100)
            predictions = self.model.predict_timing(fit_result['params'], pred_times)

            return {
                'params': fit_result['params'],
                'rmse': fit_result['rmse'],
                'aic': fit_result['aic'],
                'predictions': predictions,
                'markets': df,
                'fit_date': current_date
            }

        except Exception as e:
            # Fitting failed
            return None

    def calculate_fair_value(
        self,
        market_id: str,
        fit_result: Dict,
        current_date: pd.Timestamp
    ) -> Optional[float]:
        """
        Calculate fair value for a market based on fitted model.

        Returns:
            Fair value (probability) or None if not found
        """
        # Find this market in the fitted markets
        markets_df = fit_result['markets']
        market_row = markets_df[markets_df['market_id'] == market_id]

        if len(market_row) == 0:
            return None

        target_time_years = market_row.iloc[0]['years_to_target']

        # Get fitted CDF at this time point
        fitted_cdf = self.model._calculate_fitted_cdf(
            np.array([target_time_years]),
            fit_result['params']
        )[0]

        return fitted_cdf

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

                # Calculate fair value
                fair_value = self.calculate_fair_value(market_id, fit_result, current_date)

                if fair_value is None:
                    continue

                # Calculate edge
                edge = fair_value - market_price
                current_position = self.get_position(market_id, 'Yes')

                # BUY signal: underpriced (market < fair value)
                if edge > self.min_edge and current_position == 0:
                    signals.append(Signal(
                        market_id=market_id,
                        token_id=row['token_id'],
                        outcome='Yes',
                        signal_type=SignalType.BUY,
                        size=1.0,
                        price=market_price,
                        reason=f"Underpriced: Fair={fair_value:.3f}, Market={market_price:.3f}, Edge={edge:.3f}",
                        metadata={
                            'fair_value': fair_value,
                            'edge': edge,
                            'rmse': fit_result['rmse']
                        }
                    ))

                # SELL signal: overpriced (market > fair value)
                elif edge < -self.min_edge and current_position > 0:
                    signals.append(Signal(
                        market_id=market_id,
                        token_id=row['token_id'],
                        outcome='Yes',
                        signal_type=SignalType.SELL,
                        size=current_position,
                        price=market_price,
                        reason=f"Overpriced: Fair={fair_value:.3f}, Market={market_price:.3f}, Edge={edge:.3f}",
                        metadata={
                            'fair_value': fair_value,
                            'edge': edge,
                            'rmse': fit_result['rmse']
                        }
                    ))

        return signals

    def reset(self):
        """Reset strategy state."""
        super().reset()
        self.fitted_events = {}
        self.last_fit_date = {}
