"""
Hybrid Factored Gamma Strategy - Timing Arbitrage + Calendar Spreads

Combines two arbitrage strategies:

STRATEGY 1: Timing Distribution Arbitrage (when CDF is monotonic)
1. Build term structure from semantic group ("by April", "by May", etc.)
2. Extract implied risk rates: λ = -ln(yes_price) / time_to_expiry
3. Fit Gamma(α, β) distribution to timing CDF
4. Compare market-implied rate vs model-implied rate = rate_edge
5. BUY YES when market < model (underpriced)
6. BUY NO when market > model (overpriced)
7. EXIT when price reverts to model fair value

STRATEGY 2: Calendar Spread Arbitrage (when CDF is non-monotonic)
1. Detect violations: P(by earlier) > P(by later)
2. Identify mispriced pairs
3. BUY later-dated market (underpriced)
4. SELL earlier-dated market (overpriced)
5. Profit when spread converges or markets resolve

Position Sizing:
    Timing arb: size ∝ rate_edge
    Calendar spread: size ∝ spread_edge (CDF violation magnitude)

This hybrid approach captures both timing distribution edge AND structural
market inefficiencies from non-monotonic term structures.
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
from models.factored_gamma_model.model import FactoredGammaModel, FitResult, CalendarSpreadOpportunity
from utils.kelly_criterion import KellyCriterion


class FactoredGammaStrategy(Strategy):
    """
    Factored Gamma timing arbitrage strategy with rate edge-based position sizing.

    Fits Gamma distribution to implied timing CDFs across semantic groups,
    then trades markets that deviate from the model-implied risk rates.

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
        - min_edge: Minimum rate edge for trade entry (default: 0.10)
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
        self.calendar_spread_opportunities = 0
        self.fit_failures_by_reason = {}

        # Data loader (for loading resolved events for EB fitting)
        db_path = self.config.get('db_path', 'data/polymarket.db')
        self.data_loader = DataLoader(db_path)

    @property
    def name(self) -> str:
        """Return strategy name for logging."""
        return f"HybridGamma_TimingArb+CalendarSpread"

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
        """
        # Skip EB factor fitting - the base Gamma model works well without it
        # EB factors would provide category-specific adjustments, but require
        # significant resolved market data which is not readily available
        self.factors_fitted = True
        self.model.factors_fitted = False
        return False

    def _generate_calendar_spread_signals(
        self,
        calendar_spread: CalendarSpreadOpportunity,
        event_data: pd.DataFrame,
        current_date: pd.Timestamp,
        group_id: str
    ) -> List[Signal]:
        """
        Generate calendar spread arbitrage signals.

        Calendar spread occurs when P(by earlier) > P(by later), which is impossible.
        Arbitrage: BUY later-dated market, SELL earlier-dated market.

        If the event occurs:
        - Later market resolves YES → profit on long
        - Earlier market resolves YES → both resolve YES, spread closes to zero

        If event doesn't occur:
        - Both resolve NO → both positions profit

        Args:
            calendar_spread: CalendarSpreadOpportunity from model
            event_data: Market data for this group
            current_date: Current date
            group_id: Semantic group ID

        Returns:
            List of Signal objects for calendar spread trades
        """
        signals = []

        # Calculate total spread edge for position sizing
        total_edge = sum(pair['spread_edge'] for pair in calendar_spread.spread_pairs)

        if total_edge <= 0:
            return signals

        # Group by market_id to get both Yes and No tokens and prices
        market_tokens = {}
        market_data_by_outcome = {}  # Changed: store by (market_id, outcome)
        for _, row in event_data.iterrows():
            market_id = row['market_id']
            outcome = row['outcome']
            if market_id not in market_tokens:
                market_tokens[market_id] = {}
            market_tokens[market_id][outcome] = row['token_id']
            # Store row keyed by (market_id, outcome) to get correct prices
            market_data_by_outcome[(market_id, outcome)] = row

        # Generate signals for each spread pair
        for pair in calendar_spread.spread_pairs:
            near_market_id = pair['near_market_id']
            far_market_id = pair['far_market_id']
            spread_edge = pair['spread_edge']

            # Check if we already have positions
            near_position = self.get_position(near_market_id, 'No')
            far_position = self.get_position(far_market_id, 'Yes')

            # Skip if already in this spread
            if near_position < 0 or far_position > 0:
                continue

            # Check if these markets exist in current data (need 'No' outcome for both)
            near_key = (near_market_id, 'No')
            far_key = (far_market_id, 'No')

            if near_key not in market_data_by_outcome or far_key not in market_data_by_outcome:
                # Skip if we don't have No outcome prices for both markets
                continue

            near_row = market_data_by_outcome[near_key]
            far_row = market_data_by_outcome[far_key]

            # Calculate position size proportional to spread edge
            weight = spread_edge / total_edge
            per_spread_exposure = self.max_event_exposure * weight * 0.5  # 0.5 because we have 2 legs

            # LEG 1: SELL near-dated market (SHORT)
            # We want to short YES, which means buying NO tokens
            near_yes_price = 1.0 - near_row['price']  # Convert No price to Yes
            near_size = self._calculate_position_size(
                price=near_yes_price,
                exposure_fraction=per_spread_exposure
            )

            if near_size > 0:
                signals.append(Signal(
                    market_id=near_market_id,
                    token_id=market_tokens[near_market_id].get('No', near_row['token_id']),
                    outcome='No',  # Buy NO = Short YES
                    signal_type=SignalType.BUY,
                    size=near_size,
                    price=near_row['price'],
                    reason=f"Calendar spread (SHORT near): CDF={pair['near_cdf']:.3f} > {pair['far_cdf']:.3f}, Edge={spread_edge:.3f}",
                    metadata={
                        'spread_type': 'calendar_near_leg',
                        'spread_edge': spread_edge,
                        'event_id': group_id,
                        'pair_far_market': far_market_id
                    }
                ))

            # LEG 2: BUY far-dated market (LONG)
            far_yes_price = 1.0 - far_row['price']
            far_size = self._calculate_position_size(
                price=far_yes_price,
                exposure_fraction=per_spread_exposure
            )

            if far_size > 0:
                signals.append(Signal(
                    market_id=far_market_id,
                    token_id=market_tokens[far_market_id].get('Yes', far_row['token_id']),
                    outcome='Yes',  # Buy YES = Long
                    signal_type=SignalType.BUY,
                    size=far_size,
                    price=far_yes_price,
                    reason=f"Calendar spread (LONG far): CDF={pair['near_cdf']:.3f} > {pair['far_cdf']:.3f}, Edge={spread_edge:.3f}",
                    metadata={
                        'spread_type': 'calendar_far_leg',
                        'spread_edge': spread_edge,
                        'event_id': group_id,
                        'pair_near_market': near_market_id
                    }
                ))

        return signals

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
        Calculate position sizes based on rate edge magnitude.

        KEY IMPROVEMENT: Position sizing proportional to rate edge
            size ∝ rate_edge

        Larger rate mispricing (vs model) → larger size
        Smaller rate mispricing → smaller size

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
            # Calculate total rate edge (for normalization)
            total_edge = sum(s['rate_edge'] for s in new_position_signals)

            if total_edge > 0:
                # Allocate exposure proportional to rate edge
                for signal_info in new_position_signals:
                    row = signal_info['row']
                    prediction = signal_info['prediction']
                    rate_edge = signal_info['rate_edge']
                    outcome = signal_info['outcome']
                    direction = signal_info['direction']

                    # Weight by rate edge
                    weight = rate_edge / total_edge

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
                        f"Timing arb ({direction}): "
                        f"YesProb={1.0 - row['price']:.3f}, "
                        f"Model={prediction.median:.3f}, "
                        f"MktRate={prediction.market_implied_rate:.2f}, "
                        f"ModRate={prediction.model_implied_rate:.2f}, "
                        f"RateEdge={rate_edge:.2f}, "
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
                            'market_rate': prediction.market_implied_rate,
                            'model_rate': prediction.model_implied_rate,
                            'rate_edge': rate_edge,
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
            rate_edge = signal_info['rate_edge']

            # Get the correct price for the outcome
            if outcome == 'Yes':
                price = 1.0 - row['price']
            else:
                price = row['price']

            reason = (
                f"Exit timing arb ({direction}): "
                f"YesProb={1.0 - row['price']:.3f}, "
                f"Model={prediction.median:.3f}, "
                f"RateEdge={rate_edge:.2f}"
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
                    'market_rate': prediction.market_implied_rate,
                    'model_rate': prediction.model_implied_rate,
                    'rate_edge': rate_edge,
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
                # IMPORTANT: Only pass No outcomes to avoid duplicate term structure points
                event_data_no = event_data[event_data['outcome'] == 'No'].copy()

                self.fit_attempts += 1
                try:
                    fit_result = self.model.fit_event(event_data_no, current_date, group_id)
                    if fit_result is not None:
                        self.fitted_events[group_id] = fit_result
                        self.last_fit_date[group_id] = current_date

                        # Check if it's a calendar spread or timing arbitrage fit
                        if isinstance(fit_result, CalendarSpreadOpportunity):
                            self.calendar_spread_opportunities += 1
                        else:
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

            # Check if this is a calendar spread opportunity
            if isinstance(fit_result, CalendarSpreadOpportunity):
                # Generate calendar spread signals
                calendar_signals = self._generate_calendar_spread_signals(
                    fit_result, event_data, current_date, group_id
                )
                signals.extend(calendar_signals)
                continue  # Skip timing arbitrage logic for this group

            # PASS 1: Collect all markets that would trigger signals (timing arbitrage)
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

                # Try to get prediction from model (may be None if fit failed)
                prediction = None
                model_median = yes_price  # Default: use market price as model
                rate_edge = 0.0  # Default: no edge if model failed

                if fit_result is not None:
                    prediction = self.model.predict(market_id, fit_result, current_date)
                    if prediction is not None:
                        # Model's median prediction (in Yes probability space)
                        model_median = prediction.median
                        # Use rate edge from model (difference in implied rates)
                        rate_edge = prediction.rate_edge
                    else:
                        # Model fit succeeded but this market wasn't in the term structure
                        continue
                else:
                    # Model didn't fit for this group - skip
                    continue

                # TIMING ARBITRAGE ENTRY CRITERIA:
                # 1. Significant rate edge (market mispriced vs model)
                # 2. Model confidence (narrow CI width)
                min_rate_edge = 0.1  # Minimum 0.1 rate difference to trade
                if rate_edge < min_rate_edge:
                    continue

                current_position_no = self.get_position(market_id, 'No')
                current_position_yes = self.get_position(market_id, 'Yes')

                # Check which signal type this would trigger
                signal_info = None

                # ENTRY LOGIC: Trade based on model vs market divergence
                # If market < model median → UNDERPRICED → BUY
                # If market > model median → OVERPRICED → SELL/SHORT

                if yes_price < model_median and current_position_yes == 0:
                    # BUY YES: Market underprices event (yes_price < model)
                    signal_info = {
                        'type': SignalType.BUY,
                        'row': row,
                        'prediction': prediction,
                        'outcome': 'Yes',
                        'rate_edge': rate_edge,
                        'direction': 'arb_long_yes',
                        'market_tokens': market_tokens.get(market_id, {})
                    }
                elif yes_price > model_median and current_position_no == 0:
                    # BUY NO: Market overprices event (yes_price > model)
                    signal_info = {
                        'type': SignalType.BUY,
                        'row': row,
                        'prediction': prediction,
                        'outcome': 'No',
                        'rate_edge': rate_edge,
                        'direction': 'arb_long_no',
                        'market_tokens': market_tokens.get(market_id, {})
                    }
                # EXIT LOGIC: Price reverts to model (edge disappears)
                elif current_position_yes > 0 and yes_price >= model_median:
                    # Exit YES position: price reached or exceeded model fair value
                    signal_info = {
                        'type': SignalType.SELL,
                        'row': row,
                        'prediction': prediction,
                        'outcome': 'Yes',
                        'position': current_position_yes,
                        'rate_edge': rate_edge,
                        'direction': 'arb_exit_yes'
                    }
                elif current_position_no > 0 and yes_price <= model_median:
                    # Exit NO position: price reached or fell below model fair value
                    signal_info = {
                        'type': SignalType.SELL,
                        'row': row,
                        'prediction': prediction,
                        'outcome': 'No',
                        'position': current_position_no,
                        'rate_edge': rate_edge,
                        'direction': 'arb_exit_no'
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
        self.calendar_spread_opportunities = 0
        self.fit_failures_by_reason = {}

    def print_fit_statistics(self):
        """Print gamma fitting statistics."""
        print(f"\n{'='*70}")
        print("HYBRID STRATEGY STATISTICS")
        print(f"{'='*70}")
        print(f"Total fit attempts:              {self.fit_attempts}")
        print(f"Timing arbitrage (Gamma fits):   {self.fit_successes} ({self.fit_successes/max(self.fit_attempts,1)*100:.1f}%)")
        print(f"Calendar spread opportunities:   {self.calendar_spread_opportunities} ({self.calendar_spread_opportunities/max(self.fit_attempts,1)*100:.1f}%)")
        print(f"Failed fits:                     {self.fit_attempts - self.fit_successes - self.calendar_spread_opportunities}")

        if self.fit_failures_by_reason:
            print(f"\nFailure reasons:")
            for reason, count in sorted(self.fit_failures_by_reason.items(), key=lambda x: -x[1]):
                print(f"  - {reason}: {count}")

        print(f"\n{'='*70}")
        print("STRATEGY BREAKDOWN")
        print(f"{'='*70}")
        print(f"Timing Arbitrage:")
        print(f"  - Fits Gamma to term structure, trades rate edge")
        print(f"  - Entry: |market_rate - model_rate| > 0.1")
        print(f"  - {self.fit_successes} groups qualified")
        print(f"\nCalendar Spread Arbitrage:")
        print(f"  - Exploits non-monotonic CDFs")
        print(f"  - Entry: P(by earlier) > P(by later)")
        print(f"  - {self.calendar_spread_opportunities} groups qualified")
        print(f"{'='*70}\n")
