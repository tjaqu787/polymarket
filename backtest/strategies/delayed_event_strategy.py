"""
Delayed Event Arbitrage Strategy

Entry Logic:
- Find semantic groups with multiple resolution dates
- Short-dated market (≤20 days): No price ≥ 90% (event unlikely soon)
- Long-dated market: Yes price ≥ 80% (event likely eventually)
- BUY both: No on short-dated, Yes on long-dated

The Thesis:
Market expects event to happen BETWEEN the near and far dates.
- Collect premium on near No (event doesn't happen soon)
- Profit when event eventually happens (long Yes pays off)

Exit:
- When short-dated contract expires, close both legs
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
from datetime import datetime, timedelta

from backtest.strategy import Strategy, Signal, SignalType


class DelayedEventStrategy(Strategy):
    """
    Strategy that exploits delayed event timing signals.

    Buys:
    - Short-dated No contracts (event unlikely in near term)
    - Long-dated Yes contracts (event likely in long term)
    """

    def __init__(
        self,
        db_path: str,
        short_days_threshold: int = 20,
        short_no_threshold: float = 0.90,
        long_yes_threshold: float = 0.80,
        max_event_exposure: float = 0.15,
        **kwargs
    ):
        """
        Initialize Delayed Event Strategy.

        Args:
            db_path: Path to database
            short_days_threshold: Max days to expiry for short-dated leg (default 20)
            short_no_threshold: Min No price for short-dated leg (default 0.90)
            long_yes_threshold: Min Yes price for long-dated leg (default 0.80)
            max_event_exposure: Max portfolio exposure per event (default 15%)
        """
        super().__init__()
        self.db_path = db_path
        self.short_days_threshold = short_days_threshold
        self.short_no_threshold = short_no_threshold
        self.long_yes_threshold = long_yes_threshold
        self.max_event_exposure = max_event_exposure

        # Track statistics
        self.opportunities_found = 0
        self.trades_executed = 0

        from backtest.data_loader import DataLoader
        self.data_loader = DataLoader(db_path)

    def name(self) -> str:
        return f"DelayedEvent_No{int(self.short_no_threshold*100)}_Yes{int(self.long_yes_threshold*100)}"

    def on_data(
        self,
        current_date: pd.Timestamp,
        data: pd.DataFrame,
        portfolio_value: float = 10000.0,
        **kwargs
    ) -> List[Signal]:
        """
        Generate delayed event arbitrage signals.

        Args:
            current_date: Current date
            data: Market data for current date
            portfolio_value: Current portfolio value

        Returns:
            List of signals
        """
        signals = []

        # Group by semantic group (or event_id if no semantic grouping)
        group_col = 'group_col' if 'group_col' in data.columns else 'event_id'

        for group_id in data[group_col].unique():
            event_data = data[data[group_col] == group_id]

            # Need at least 2 markets with different resolution dates
            unique_resolutions = event_data['resolution_date'].unique()
            if len(unique_resolutions) < 2:
                continue

            # Calculate time to expiry for each market
            event_data = event_data.copy()
            event_data['days_to_expiry'] = (
                event_data['resolution_date'] - current_date
            ).dt.days

            # Find short-dated No candidates (≤20 days, No ≥ 90%)
            short_candidates = event_data[
                (event_data['days_to_expiry'] <= self.short_days_threshold) &
                (event_data['days_to_expiry'] > 0) &
                (event_data['outcome'] == 'No') &
                (event_data['price'] >= self.short_no_threshold)
            ]

            if short_candidates.empty:
                continue

            # Find long-dated Yes candidates (Yes ≥ 80%, further out than short)
            long_candidates = event_data[
                (event_data['outcome'] == 'Yes')
            ].copy()

            # For each short candidate, find matching long candidate
            for _, short_row in short_candidates.iterrows():
                short_market_id = short_row['market_id']
                short_days = short_row['days_to_expiry']
                short_resolution = short_row['resolution_date']

                # Find long-dated candidates
                # (further out than short, Yes ≥ 80%)
                matching_longs = long_candidates[
                    (long_candidates['resolution_date'] > short_resolution) &
                    (long_candidates['price'] >= self.long_yes_threshold)
                ]

                if matching_longs.empty:
                    continue

                # Pick the closest long-dated market
                long_row = matching_longs.iloc[0]
                long_market_id = long_row['market_id']
                long_yes_price = long_row['price']

                # Found an opportunity!
                self.opportunities_found += 1

                # Calculate position size (split exposure 50/50 between legs)
                per_leg_exposure = self.max_event_exposure * 0.5
                short_size = self._calculate_position_size(
                    price=short_row['price'],
                    exposure_fraction=per_leg_exposure,
                    portfolio_value=portfolio_value
                )
                long_size = self._calculate_position_size(
                    price=long_yes_price,
                    exposure_fraction=per_leg_exposure,
                    portfolio_value=portfolio_value
                )

                if short_size > 0 and long_size > 0:
                    # LEG 1: BUY short-dated No
                    signals.append(Signal(
                        market_id=short_market_id,
                        token_id=short_row['token_id'],
                        outcome='No',
                        signal_type=SignalType.BUY,
                        size=short_size,
                        price=short_row['price'],
                        reason=f"Delayed event SHORT leg: No={short_row['price']:.2f}, {short_days}d to expiry",
                        metadata={
                            'strategy_type': 'delayed_event',
                            'spread_type': 'delayed_short_leg',
                            'pair_long_market': long_market_id,
                            'event_id': group_id
                        }
                    ))

                    # LEG 2: BUY long-dated Yes
                    signals.append(Signal(
                        market_id=long_market_id,
                        token_id=long_row['token_id'],
                        outcome='Yes',
                        signal_type=SignalType.BUY,
                        size=long_size,
                        price=long_yes_price,
                        reason=f"Delayed event LONG leg: Yes={long_yes_price:.2f}, {long_row['days_to_expiry']}d to expiry",
                        metadata={
                            'strategy_type': 'delayed_event',
                            'spread_type': 'delayed_long_leg',
                            'pair_short_market': short_market_id,
                            'event_id': group_id
                        }
                    ))

                    self.trades_executed += 1

                # Only take first opportunity per group
                break

        return signals

    def _calculate_position_size(
        self, price: float, exposure_fraction: float, portfolio_value: float
    ) -> float:
        """
        Calculate position size based on exposure fraction.

        Args:
            price: Entry price per contract
            exposure_fraction: Fraction of portfolio to allocate
            portfolio_value: Current portfolio value

        Returns:
            Number of contracts to buy
        """
        if price <= 0 or portfolio_value <= 0:
            return 0.0

        dollar_exposure = portfolio_value * exposure_fraction
        size = dollar_exposure / price

        return round(size)

    def on_backtest_end(self, **kwargs):
        """Print strategy statistics."""
        print(f"\n{'='*70}")
        print("DELAYED EVENT STRATEGY STATISTICS")
        print(f"{'='*70}")
        print(f"Opportunities found:             {self.opportunities_found}")
        print(f"Pairs traded:                    {self.trades_executed}")
        print(f"\nStrategy Parameters:")
        print(f"  Short-dated: ≤{self.short_days_threshold} days, No ≥ {self.short_no_threshold*100:.0f}%")
        print(f"  Long-dated: Yes ≥ {self.long_yes_threshold*100:.0f}%")
        print(f"  Max event exposure: {self.max_event_exposure*100:.0f}%")
        print(f"{'='*70}\n")
