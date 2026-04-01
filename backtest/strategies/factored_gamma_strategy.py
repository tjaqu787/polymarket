"""
Factored Gamma Carry Strategy

Carry trading strategy using the Factored Gamma Model for prediction markets.
Trades extreme probability markets near expiry, scaling positions by gamma edge.

Trading Logic:
- BUY YES when Yes_prob > 0.90 AND TTE ≤ 30 days (high confidence events)
- BUY NO when Yes_prob < 0.10 AND TTE ≤ 30 days (low confidence events)
- EXIT when TTE ≤ 7 days (near expiry)

Position Sizing:
    size ∝ gamma_edge * max_event_exposure

Larger gamma edge (|market - model|) → larger position
Smaller gamma edge → smaller position

The gamma model fits a term structure across semantic groups to identify
markets that are mispriced relative to the implied volatility surface.
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
from utils.kelly_criterion import KellyCriterion


class FactoredGammaStrategy(Strategy):
    """
    Factored Gamma carry trading strategy with gamma edge-based position sizing.

    Trades extreme probability markets (< 0.10 or > 0.90) near expiry (TTE ≤ 30 days),
    using a Gamma model fitted across semantic groups to identify mispriced markets.

    Configuration Parameters:
        - db_path: Path to polymarket.db (default: 'data/polymarket.db')
        - min_buckets: Min markets in semantic group (default: 3)
        - max_rmse: Max fit error threshold (default: 0.3)
        - ci_level: Credible interval level (default: 0.70)
        - n_bootstrap: Bootstrap samples (default: 500)
        - refit_hours: Hours between refits (default: 6)
        - max_event_exposure: Max portfolio % per event (default: 0.15)
        - eb_holdout_end_date: Factor estimation cutoff (default: "2025-10-05")
        - use_kelly_sizing: Use Kelly criterion for position sizing (default: True)
        - kelly_fraction: Fractional Kelly to use (default: 0.25)
        - min_edge: Minimum edge for Kelly (default: 0.05)
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the Factored Gamma strategy."""
        super().__init__(config)

        # Extract config parameters
        self.min_buckets = self.config.get('min_buckets', 3)
        self.max_rmse = self.config.get('max_rmse', 0.3)
        self.ci_level = self.config.get('ci_level', 0.70)
        self.n_bootstrap = self.config.get('n_bootstrap', 500)
        self.refit_hours = self.config.get('refit_hours', 6)  # Changed from refit_days to refit_hours
        self.max_event_exposure = self.config.get('max_event_exposure', 0.15)
        self.eb_holdout_end_date = self.config.get('eb_holdout_end_date', '2025-10-05')

        # Kelly criterion parameters
        self.use_kelly_sizing = self.config.get('use_kelly_sizing', True)
        self.kelly_fraction = self.config.get('kelly_fraction', 0.25)
        self.min_edge = self.config.get('min_edge', 0.05)

        # Initialize Kelly criterion
        if self.use_kelly_sizing:
            self.kelly_calculator = KellyCriterion(
                kelly_fraction=self.kelly_fraction,
                min_edge=self.min_edge,
                max_position=self.max_event_exposure,
                carry_penalty=0.5
            )
        else:
            self.kelly_calculator = None

        # Initialize model
        self.model = FactoredGammaModel(
            min_buckets=self.min_buckets,
            max_rmse=self.max_rmse,
            ci_level=self.ci_level,
            n_bootstrap=self.n_bootstrap
        )

        # Track fitted events
        self.fitted_events = {}  # event_id -> FitResult (or None if fit failed)
        self.last_fit_date = {}  # event_id -> date
        self.factors_fitted = False

        # Track fitting statistics
        self.fit_attempts = 0
        self.fit_successes = 0
        self.fit_failures_by_reason = {}

        # Data loader (for loading resolved events for EB fitting)
        db_path = self.config.get('db_path', 'data/polymarket.db')
        self.data_loader = DataLoader(db_path)

    @property
    def name(self) -> str:
        """Return strategy name for logging."""
        return f"FactoredGammaCarry_TTE30"

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
        """
        Check if an event has enough markets for gamma fitting.

        For carry trading, we need markets at different resolutions dates
        or different thresholds within the same semantic group.
        """
        # Count unique markets in this group
        num_markets = event_data['market_id'].nunique()
        return num_markets >= self.min_buckets

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
        Calculate position sizes based on gamma edge magnitude.

        KEY IMPROVEMENT: Position sizing proportional to gamma edge
            size ∝ gamma_edge

        Larger mispricing (vs model) → larger size
        Smaller mispricing → smaller size

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
            if s['type'] == SignalType.BUY and 'position' not in s
        ]

        # Calculate position sizes for new positions
        if len(new_position_signals) > 0:
            # Calculate total gamma edge (for normalization)
            total_edge = sum(s['gamma_edge'] for s in new_position_signals)

            if total_edge > 0:
                # Allocate exposure proportional to gamma edge
                for signal_info in new_position_signals:
                    row = signal_info['row']
                    prediction = signal_info['prediction']
                    gamma_edge = signal_info['gamma_edge']
                    outcome = signal_info['outcome']
                    direction = signal_info['direction']

                    # Weight by gamma edge
                    weight = gamma_edge / total_edge

                    # Allocate fraction of max_event_exposure
                    per_market_exposure = self.max_event_exposure * weight

                    # Get the correct price for the outcome we're trading
                    if outcome == 'Yes':
                        price = 1.0 - row['price']  # Convert No price to Yes price
                    else:
                        price = row['price']  # Use No price directly

                    # Calculate size
                    size = self._calculate_position_size(
                        price=price,
                        exposure_fraction=per_market_exposure
                    )

                    # Get token_id for the outcome we're trading
                    market_tokens = signal_info.get('market_tokens', {})
                    token_id = market_tokens.get(outcome, row['token_id'])

                    reason = (
                        f"Carry trade ({direction}): "
                        f"YesProb={1.0 - row['price']:.3f}, "
                        f"Model={prediction.median:.3f}, "
                        f"Edge={gamma_edge:.3f}, "
                        f"Allocation={per_market_exposure*100:.1f}%"
                    )

                    signals.append(Signal(
                        market_id=row['market_id'],
                        token_id=token_id,
                        outcome=outcome,
                        signal_type=SignalType.BUY,
                        size=size,
                        price=price,
                        reason=reason,
                        metadata={
                            'model_median': prediction.median,
                            'gamma_edge': gamma_edge,
                            'interval_width': prediction.interval_width,
                            'alpha_adjusted': prediction.alpha_adjusted,
                            'beta_adjusted': prediction.beta_adjusted,
                            'event_exposure': per_market_exposure,
                            'event_id': event_id,
                            'direction': direction
                        }
                    ))

        # Process exit signals (SELL)
        exit_signals = [
            s for s in potential_signals
            if s['type'] == SignalType.SELL
        ]

        for signal_info in exit_signals:
            row = signal_info['row']
            prediction = signal_info['prediction']
            outcome = signal_info['outcome']
            position_size = signal_info['position']
            direction = signal_info['direction']

            # Get the correct price for the outcome
            if outcome == 'Yes':
                price = 1.0 - row['price']
            else:
                price = row['price']

            reason = (
                f"Exit carry trade ({direction}): "
                f"YesProb={1.0 - row['price']:.3f}, "
                f"Model={prediction.median:.3f}"
            )

            signals.append(Signal(
                market_id=row['market_id'],
                token_id=row['token_id'],
                outcome=outcome,
                signal_type=SignalType.SELL,
                size=position_size,
                price=price,
                reason=reason,
                metadata={
                    'model_median': prediction.median,
                    'event_id': event_id,
                    'direction': direction
                }
            ))

        return signals

    def on_data(self, current_date: pd.Timestamp, data: pd.DataFrame) -> List[Signal]:
        """
        Generate trading signals based on Factored Gamma model.

        Implementation:
            1. FIRST CALL ONLY: Fit Empirical Bayes factors (if enabled)
            2. EVERY TIMESTEP:
                - Group data by event
                - Filter to time-distributed events
                - Every refit_hours: Call model.fit_event() (refits from scratch via MLE)
                - For each market: Get prediction, generate signals
            3. Calculate position sizes using interval_width

        Note on Bayesian Inference:
            Current implementation uses MLE + bootstrap (frequentist approach):
            - Each refit estimates α, β from scratch via maximum likelihood
            - Bootstrap resampling provides credible intervals
            - NOT sequential Bayesian updating (no prior→posterior chaining)

            For true Bayesian refitting, would need:
            - PyMC model with priors on (α, β)
            - Use previous fit's posterior as new prior when refitting
            - MCMC sampling for posterior credible intervals
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

            # For each market, keep only the latest price snapshot of the day
            # This avoids having 24 hourly snapshots creating duplicate term structure points
            if 'ts' in event_data.columns:
                event_data = event_data.sort_values('ts').groupby(['market_id', 'outcome'], as_index=False).last()

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
                # Fit event using Factored Gamma Model
                self.fit_attempts += 1
                try:
                    fit_result = self.model.fit_event(event_data, current_date, group_id)
                    if fit_result is not None:
                        self.fitted_events[group_id] = fit_result
                        self.last_fit_date[group_id] = current_date
                        self.fit_successes += 1
                    else:
                        # Fitting failed - store None to indicate we tried
                        # We'll still trade but without gamma edge sizing
                        self.fitted_events[group_id] = None
                        self.last_fit_date[group_id] = current_date
                        reason = "Unknown"
                        self.fit_failures_by_reason[reason] = self.fit_failures_by_reason.get(reason, 0) + 1
                except Exception as e:
                    # Capture the failure reason
                    reason = str(e).split(':')[0] if ':' in str(e) else str(e)[:50]
                    self.fit_failures_by_reason[reason] = self.fit_failures_by_reason.get(reason, 0) + 1
                    self.fitted_events[group_id] = None
                    self.last_fit_date[group_id] = current_date

            # Get fitted model for this event (may be None if fitting failed)
            fit_result = self.fitted_events.get(group_id, None)

            # PASS 1: Collect all markets that would trigger signals
            potential_signals = []

            # Group by market_id to get both Yes and No tokens
            market_tokens = {}
            for _, row in event_data.iterrows():
                market_id = row['market_id']
                outcome = row['outcome']
                if market_id not in market_tokens:
                    market_tokens[market_id] = {}
                market_tokens[market_id][outcome] = row['token_id']

            # Iterate over No outcome rows (we'll use them as reference)
            for _, row in event_data[event_data['outcome'] == 'No'].iterrows():
                market_id = row['market_id']
                no_price = row['price']
                yes_price = 1.0 - no_price

                # Check if we have time_to_expiration column
                if 'time_to_expiration' not in row.index:
                    # Calculate TTE if not present
                    if 'resolution_date' in row.index:
                        tte_days = (pd.to_datetime(row['resolution_date']) - current_date).days
                        tte_years = tte_days / 365.25
                    else:
                        continue
                else:
                    tte_years = row['time_to_expiration']

                # CARRY TRADE ENTRY CRITERIA:
                # 1. TTE <= 30 days (0.082 years)
                # 2. Extreme probability (Yes < 0.10 or Yes > 0.90)
                if tte_years > 30.0 / 365.25:
                    continue

                # Try to get prediction from model (may be None if fit failed)
                prediction = None
                model_median = yes_price  # Default: use market price as model
                gamma_edge = 0.05  # Default edge for equal weighting

                if fit_result is not None:
                    prediction = self.model.predict(market_id, fit_result, current_date)
                    if prediction is not None:
                        # Model's median prediction (in Yes probability space)
                        model_median = prediction.median
                        # Calculate gamma edge (how far actual price is from model)
                        gamma_edge = abs(yes_price - model_median)

                current_position_no = self.get_position(market_id, 'No')
                current_position_yes = self.get_position(market_id, 'Yes')

                # Check which signal type this would trigger
                signal_info = None

                # ENTRY LOGIC: Buy Yes when prob > 0.90, No when prob < 0.10
                if yes_price > 0.90 and current_position_yes == 0:
                    # Create prediction if model didn't fit
                    if prediction is None:
                        from models.factored_gamma_model.model import PredictionResult
                        prediction = PredictionResult(
                            market_id=market_id,
                            lower_bound=yes_price - 0.05,
                            upper_bound=yes_price + 0.05,
                            median=yes_price,
                            interval_width=0.10,
                            alpha_adjusted=1.0,
                            beta_adjusted=1.0,
                            time_to_target=tte_years
                        )
                    # BUY YES: High probability event, bet with consensus
                    signal_info = {
                        'type': SignalType.BUY,
                        'row': row,
                        'prediction': prediction,
                        'outcome': 'Yes',
                        'gamma_edge': gamma_edge,
                        'direction': 'long_yes',
                        'market_tokens': market_tokens.get(market_id, {})
                    }
                elif yes_price < 0.10 and current_position_no == 0:
                    if prediction is None:
                        from models.factored_gamma_model.model import PredictionResult
                        prediction = PredictionResult(
                            market_id=market_id,
                            lower_bound=yes_price - 0.05,
                            upper_bound=yes_price + 0.05,
                            median=yes_price,
                            interval_width=0.10,
                            alpha_adjusted=1.0,
                            beta_adjusted=1.0,
                            time_to_target=tte_years
                        )
                    # BUY NO: Low probability event, bet against consensus
                    signal_info = {
                        'type': SignalType.BUY,
                        'row': row,
                        'prediction': prediction,
                        'outcome': 'No',
                        'gamma_edge': gamma_edge,
                        'direction': 'long_no',
                        'market_tokens': market_tokens.get(market_id, {})
                    }
                # EXIT LOGIC: Exit at TTE = 7 days or if probability normalizes
                elif tte_years <= 7.0 / 365.25:
                    # Exit near expiry
                    if current_position_yes > 0:
                        # Create a fake prediction for exit signals if model didn't fit
                        if prediction is None:
                            from models.factored_gamma_model.model import PredictionResult
                            prediction = PredictionResult(
                                market_id=market_id,
                                lower_bound=yes_price - 0.05,
                                upper_bound=yes_price + 0.05,
                                median=yes_price,
                                interval_width=0.10,
                                alpha_adjusted=1.0,
                                beta_adjusted=1.0,
                                time_to_target=tte_years
                            )
                        signal_info = {
                            'type': SignalType.SELL,
                            'row': row,
                            'prediction': prediction,
                            'outcome': 'Yes',
                            'position': current_position_yes,
                            'gamma_edge': gamma_edge,
                            'direction': 'exit_yes'
                        }
                    elif current_position_no > 0:
                        if prediction is None:
                            from models.factored_gamma_model.model import PredictionResult
                            prediction = PredictionResult(
                                market_id=market_id,
                                lower_bound=yes_price - 0.05,
                                upper_bound=yes_price + 0.05,
                                median=yes_price,
                                interval_width=0.10,
                                alpha_adjusted=1.0,
                                beta_adjusted=1.0,
                                time_to_target=tte_years
                            )
                        signal_info = {
                            'type': SignalType.SELL,
                            'row': row,
                            'prediction': prediction,
                            'outcome': 'No',
                            'position': current_position_no,
                            'gamma_edge': gamma_edge,
                            'direction': 'exit_no'
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
        self.fit_attempts = 0
        self.fit_successes = 0
        self.fit_failures_by_reason = {}

    def print_fit_statistics(self):
        """Print gamma fitting statistics."""
        print(f"\n{'='*70}")
        print("GAMMA FITTING STATISTICS")
        print(f"{'='*70}")
        print(f"Total fit attempts:    {self.fit_attempts}")
        print(f"Successful fits:       {self.fit_successes} ({self.fit_successes/max(self.fit_attempts,1)*100:.1f}%)")
        print(f"Failed fits:           {self.fit_attempts - self.fit_successes}")

        if self.fit_failures_by_reason:
            print(f"\nFailure reasons:")
            for reason, count in sorted(self.fit_failures_by_reason.items(), key=lambda x: -x[1]):
                print(f"  - {reason}: {count}")

        print(f"\nNote: Non-monotonic CDFs may indicate arbitrage opportunities")
        print(f"      (calendar spreads where near-dated > far-dated contracts)")
        print(f"{'='*70}\n")
