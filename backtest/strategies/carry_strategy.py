"""
Carry Strategy for markets near expiry with extreme probabilities.

This strategy captures the carry (time decay) for markets close to expiry
where the probability is very high (>0.90) or very low (<0.10).
"""

import pandas as pd
from typing import List, Dict, Optional
from datetime import datetime, timedelta

from backtest.strategy import Strategy, Signal, SignalType


class CarryStrategy(Strategy):
    """
    Carry strategy for markets near expiry with extreme probabilities.

    Strategy logic:
    1. Identify markets < 90 days to expiry
    2. BUY (long Yes) when price > 0.90 (expecting it to reach 1.00)
    3. SHORT (sell Yes/buy No) when price < 0.10 (expecting it to reach 0.00)
    4. Hold until expiry to capture the carry
    5. Exit if price moves against us (stop loss)
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)

        # Strategy parameters
        self.max_days_to_expiry = self.config.get('max_days_to_expiry', 90)
        self.high_threshold = self.config.get('high_threshold', 0.90)
        self.low_threshold = self.config.get('low_threshold', 0.10)

        # Stop loss parameters
        self.stop_loss_long = self.config.get('stop_loss_long', 0.85)  # Exit long if drops below
        self.stop_loss_short = self.config.get('stop_loss_short', 0.15)  # Exit short if rises above

        # Position tracking
        self.entry_prices = {}  # (market_id, outcome) -> entry_price

    @property
    def name(self) -> str:
        return f"Carry_DTE{self.max_days_to_expiry}_H{int(self.high_threshold*100)}_L{int(self.low_threshold*100)}"

    def calculate_days_to_expiry(self, end_date: pd.Timestamp, current_date: pd.Timestamp) -> int:
        """Calculate days remaining until market expiry."""
        if pd.isna(end_date):
            return float('inf')
        return (end_date - current_date).days

    def on_data(self, current_date: pd.Timestamp, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals for carry opportunities."""
        signals = []

        # Get data for current date
        today_data = data[data['date'] == current_date]

        if today_data.empty:
            return signals

        # Process each market
        for _, row in today_data.iterrows():
            if row['outcome'] != 'Yes':
                continue

            market_id = row['market_id']
            token_id = row['token_id']
            price = row['price']
            end_date = row.get('end_date')

            # Check if we have an end date
            if pd.isna(end_date):
                continue

            # Calculate days to expiry
            days_to_expiry = self.calculate_days_to_expiry(end_date, current_date)

            # Skip if too far from expiry or already expired
            if days_to_expiry > self.max_days_to_expiry or days_to_expiry <= 0:
                continue

            # Get current position
            position = self.get_position(market_id, 'Yes')
            position_key = (market_id, 'Yes')

            # === ENTRY SIGNALS ===

            # Long signal: High probability market (expecting to go to 1.00)
            if price > self.high_threshold and position == 0:
                signals.append(Signal(
                    market_id=market_id,
                    token_id=token_id,
                    outcome='Yes',
                    signal_type=SignalType.BUY,
                    size=1.0,
                    price=price,
                    reason=f"Carry long: Price={price:.3f}, DTE={days_to_expiry}, expecting → 1.00",
                    metadata={
                        'days_to_expiry': days_to_expiry,
                        'expected_return': 1.0 - price,
                        'strategy_type': 'carry_long'
                    }
                ))
                self.entry_prices[position_key] = price

            # Short signal: Low probability market (expecting to go to 0.00)
            # Note: In practice, this would be shorting Yes or buying No
            elif price < self.low_threshold and position == 0:
                # For simplicity, we'll track this as a negative position on Yes
                # In real implementation, this would be buying No tokens
                signals.append(Signal(
                    market_id=market_id,
                    token_id=token_id,
                    outcome='Yes',
                    signal_type=SignalType.SELL,  # Selling Yes = shorting
                    size=1.0,
                    price=price,
                    reason=f"Carry short: Price={price:.3f}, DTE={days_to_expiry}, expecting → 0.00",
                    metadata={
                        'days_to_expiry': days_to_expiry,
                        'expected_return': price,  # We profit from price going to 0
                        'strategy_type': 'carry_short'
                    }
                ))
                self.entry_prices[position_key] = price

            # === EXIT SIGNALS ===

            # Stop loss for long positions
            elif position > 0:
                entry_price = self.entry_prices.get(position_key, price)

                # Exit if price drops below stop loss
                if price < self.stop_loss_long:
                    signals.append(Signal(
                        market_id=market_id,
                        token_id=token_id,
                        outcome='Yes',
                        signal_type=SignalType.SELL,
                        size=position,
                        price=price,
                        reason=f"Stop loss long: Price={price:.3f} < {self.stop_loss_long}, Entry={entry_price:.3f}",
                        metadata={
                            'entry_price': entry_price,
                            'exit_reason': 'stop_loss'
                        }
                    ))
                    if position_key in self.entry_prices:
                        del self.entry_prices[position_key]

            # Stop loss for short positions (tracked as negative positions)
            elif position < 0:
                entry_price = self.entry_prices.get(position_key, price)

                # Exit if price rises above stop loss
                if price > self.stop_loss_short:
                    signals.append(Signal(
                        market_id=market_id,
                        token_id=token_id,
                        outcome='Yes',
                        signal_type=SignalType.BUY,  # Buy back to close short
                        size=abs(position),
                        price=price,
                        reason=f"Stop loss short: Price={price:.3f} > {self.stop_loss_short}, Entry={entry_price:.3f}",
                        metadata={
                            'entry_price': entry_price,
                            'exit_reason': 'stop_loss'
                        }
                    ))
                    if position_key in self.entry_prices:
                        del self.entry_prices[position_key]

        return signals

    def on_market_close(self, market_id: str, outcome: str, final_price: float):
        """Clean up tracking when market closes."""
        super().on_market_close(market_id, outcome, final_price)

        # Remove entry price tracking
        position_key = (market_id, outcome)
        if position_key in self.entry_prices:
            del self.entry_prices[position_key]

    def reset(self):
        """Reset strategy state."""
        super().reset()
        self.entry_prices = {}
