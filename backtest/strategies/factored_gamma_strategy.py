"""
Factored Gamma Timing Strategy

Trading strategy using the Factored Gamma Model for prediction market timing.
Unifies PoissonTimingStrategy with Empirical Bayes factor adjustments.

Trading Logic:
- BUY when market_price < CI_lower (underpriced relative to timing model)
- SHORT when market_price > CI_upper (overpriced)
- SELL when long position price reaches CI_upper
- COVER when short position price reaches CI_lower

Position Sizing:
    size = max_event_exposure / (num_signals * interval_width)

Narrow CI (confident timing) → larger size
Wide CI (uncertain timing) → smaller size
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
from models.factored_gamma_model.model import FactoredGammaModel


class FactoredGammaStrategy(Strategy):
    """
    Factored Gamma timing strategy with dynamic position sizing.

    Replaces both TimeDiscountingStrategy and PoissonTimingStrategy
    by combining Gamma CDF fitting with Empirical Bayes factor adjustments.

    Configuration Parameters:
        - db_path: Path to polymarket.db (default: 'data/polymarket.db')
        - min_buckets: Min term structure points (default: 3)
        - max_rmse: Max fit error threshold (default: 0.3)
        - ci_level: Credible interval level (default: 0.70)
        - n_bootstrap: Bootstrap samples (default: 500)
        - refit_days: Days between refits (default: 7)
        - max_event_exposure: Max portfolio % per event (default: 0.15)
        - eb_holdout_end_date: Factor estimation cutoff (default: "2025-10-05")
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Factored Gamma strategy."""
        super().__init__(config)

        # Extract config parameters
        self.min_buckets = self.config.get('min_buckets', 3)
        self.max_rmse = self.config.get('max_rmse', 0.3)
        self.ci_level = self.config.get('ci_level', 0.70)
        self.n_bootstrap = self.config.get('n_bootstrap', 500)
        self.refit_days = self.config.get('refit_days', 7)
        self.max_event_exposure = self.config.get('max_event_exposure', 0.15)
        self.eb_holdout_end_date = self.config.get('eb_holdout_end_date', '2025-10-05')

        # Initialize model
        self.model = FactoredGammaModel(
            min_buckets=self.min_buckets,
            max_rmse=self.max_rmse,
            ci_level=self.ci_level,
            n_bootstrap=self.n_bootstrap
        )

        # Track fitted events
        self.fitted_events = {}  # event_id -> FitResult
        self.last_fit_date = {}  # event_id -> date
        self.factors_fitted = False

        # Data loader (for loading resolved events for EB fitting)
        db_path = self.config.get('db_path', 'data/polymarket.db')
        self.data_loader = DataLoader(db_path)

    @property
    def name(self) -> str:
        """Return strategy name for logging."""
        return f"FactoredGamma_CI{int(self.ci_level*100)}"

    def convert_no_price_to_yes(self, no_price: float) -> float:
        """
        Convert 'No' outcome price to equivalent 'Yes' price.

        Since P(No) = 1 - P(Yes), we convert No prices to Yes prices.
        """
        return 1.0 - no_price

    def convert_yes_bounds_to_no(
        self, yes_lower: float, yes_upper: float, yes_median: float
    ) -> Dict[str, float]:
        """
        Convert 'Yes' outcome CI bounds to 'No' outcome bounds.

        Since P(No) = 1 - P(Yes), bounds are inverted and swapped:
        - No lower bound = 1 - Yes upper bound
        - No upper bound = 1 - Yes lower bound
        """
        return {
            'lower': 1.0 - yes_upper,   # No lower = 1 - Yes upper (bounds swap!)
            'upper': 1.0 - yes_lower,   # No upper = 1 - Yes lower
            'median': 1.0 - yes_median
        }

    def extract_target_date(self, question: str) -> Optional[datetime]:
        """Extract target date from question text."""
        patterns = [
            r'by ([A-Za-z]+) (\d+),? (\d{4})',  # "by March 15, 2026" or "by March 15 2026"
            r'before ([A-Za-z]+) (\d+),? (\d{4})',  # "before March 15, 2026"
            r'no later than ([A-Za-z]+) (\d+),? (\d{4})',  # "no later than March 15, 2026"
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
                        # Try abbreviated month
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

    def fit_empirical_bayes_factors(self, current_date: str) -> bool:
        """
        Fit empirical Bayes factors on holdout data.

        Called once at strategy initialization (first on_data() call).

        Args:
            current_date: Current backtest date

        Returns:
            True if fitting succeeded, False otherwise

        Note:
            This is a simplified implementation. In production, you would:
            1. Load resolved markets from database
            2. Extract term structure snapshots
            3. Call model.fit_factors()

            For now, we skip EB fitting and use base parameters only.
            This can be implemented when resolved events data is properly structured.
        """
        print(f"\n{'='*70}")
        print("FACTORED GAMMA STRATEGY: Fitting Empirical Bayes Factors")
        print(f"{'='*70}")
        print(f"Holdout end date: {self.eb_holdout_end_date}")
        print(f"Current date: {current_date}")

        # TODO: Implement resolved events loading and factor fitting
        # For now, skip EB fitting (factors_fitted will remain False)
        # The model will use base parameters without factor adjustments

        print("\nWARNING: Empirical Bayes factor fitting not yet implemented")
        print("Using base Gamma parameters without category adjustments")
        print("To enable EB factors, implement resolved events data loading\n")

        # Set flag even though we didn't fit (prevents repeated attempts)
        self.factors_fitted = True

        return False  # Indicate EB fitting not actually performed

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
        event_id: str
    ) -> List[Signal]:
        """
        Calculate position sizes based on interval width.

        KEY IMPROVEMENT: Position sizing inversely proportional to interval width
            size ∝ 1 / interval_width

        Narrow CI (confident) → larger size
        Wide CI (uncertain) → smaller size

        Args:
            potential_signals: List of signal dicts from first pass
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

            # Allocate exposure proportional to confidence (inverse width)
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
                    f"{reason_prefix} {int(self.ci_level*100)}% CI: "
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
                        'alpha_adjusted': prediction.alpha_adjusted,
                        'beta_adjusted': prediction.beta_adjusted,
                        'event_exposure': per_market_exposure,
                        'event_id': event_id
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
                f"{reason_prefix} at {bound_type} bound: "
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
                    'event_id': event_id
                }
            ))

        return signals

    def on_data(self, current_date: pd.Timestamp, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on Factored Gamma model.

        Implementation:
            1. FIRST CALL ONLY: Fit Empirical Bayes factors (if enabled)
            2. EVERY DAY:
                - Group data by event
                - Filter to time-distributed events
                - Every refit_days: Call model.fit_event()
                - For each market: Get prediction, generate signals
            3. Calculate position sizes using interval_width
        """
        signals = []

        # INITIALIZATION: Fit EB factors on first call
        if not self.factors_fitted:
            self.fit_empirical_bayes_factors(str(current_date))

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
                (current_date - self.last_fit_date[group_id]).days >= self.refit_days
            )

            if need_fit:
                # Fit event using Factored Gamma Model
                fit_result = self.model.fit_event(event_data, current_date, group_id)

                if fit_result is not None:
                    self.fitted_events[group_id] = fit_result
                    self.last_fit_date[group_id] = current_date
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
        self.factors_fitted = False
