"""
Base Strategy class for backtesting.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
from dataclasses import dataclass
from enum import Enum


class SignalType(Enum):
    """Trading signal types."""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


@dataclass
class Signal:
    """Trading signal."""
    market_id: str
    token_id: str
    outcome: str
    signal_type: SignalType
    size: float = 1.0  # Position size (fraction of capital or number of contracts)
    price: Optional[float] = None
    reason: Optional[str] = None
    metadata: Optional[Dict] = None


class Strategy(ABC):
    """
    Base class for trading strategies.

    Subclasses must implement:
    - on_data(): Process new market data and generate signals
    - name: Property returning strategy name
    """

    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize strategy.

        Args:
            config: Strategy configuration parameters
        """
        self.config = config or {}
        self.positions = {}  # Current positions
        self.signals = []  # Historical signals

    @property
    @abstractmethod
    def name(self) -> str:
        """Return strategy name."""
        pass

    @abstractmethod
    def on_data(self, current_date: pd.Timestamp, data: pd.DataFrame) -> List[Signal]:
        """
        Process new data and generate trading signals.

        Args:
            current_date: Current timestamp in backtest
            data: Market data up to current_date (historical data)

        Returns:
            List of trading signals
        """
        pass

    def on_market_close(self, market_id: str, outcome: str, final_price: float):
        """
        Called when a market closes/resolves.

        Args:
            market_id: Market that closed
            outcome: Winning outcome
            final_price: Final settlement price (typically 0 or 1)
        """
        pass

    def reset(self):
        """Reset strategy state."""
        self.positions = {}
        self.signals = []

    def get_position(self, market_id: str, outcome: str) -> float:
        """
        Get current position size for a market.

        Args:
            market_id: Market ID
            outcome: Outcome (Yes/No)

        Returns:
            Position size (0 if no position)
        """
        key = (market_id, outcome)
        return self.positions.get(key, 0.0)

    def update_position(self, market_id: str, outcome: str, size: float):
        """
        Update position for a market.

        Args:
            market_id: Market ID
            outcome: Outcome
            size: New position size
        """
        key = (market_id, outcome)
        if size == 0 and key in self.positions:
            del self.positions[key]
        else:
            self.positions[key] = size


class BuyAndHoldStrategy(Strategy):
    """
    Simple buy-and-hold strategy.
    Buys Yes outcome and holds until expiration.
    """

    @property
    def name(self) -> str:
        return "BuyAndHold"

    def on_data(self, current_date: pd.Timestamp, data: pd.DataFrame) -> List[Signal]:
        """Buy all available markets once at first opportunity."""
        signals = []

        # Get data for current date
        today_data = data[data['date'] == current_date]

        if today_data.empty:
            return signals

        # For each market, buy if we don't have a position
        for market_id in today_data['market_id'].unique():
            market_data = today_data[
                (today_data['market_id'] == market_id) &
                (today_data['outcome'] == 'Yes')
            ]

            if market_data.empty:
                continue

            # Check if we already have a position
            if self.get_position(market_id, 'Yes') > 0:
                continue

            # Create buy signal
            row = market_data.iloc[0]
            signals.append(Signal(
                market_id=market_id,
                token_id=row['token_id'],
                outcome='Yes',
                signal_type=SignalType.BUY,
                size=1.0,
                price=row['price'],
                reason="Buy and hold"
            ))

        return signals


class ThresholdStrategy(Strategy):
    """
    Buy when price below threshold, sell when above threshold.
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)
        self.buy_threshold = self.config.get('buy_threshold', 0.4)
        self.sell_threshold = self.config.get('sell_threshold', 0.6)

    @property
    def name(self) -> str:
        return f"Threshold_{self.buy_threshold}_{self.sell_threshold}"

    def on_data(self, current_date: pd.Timestamp, data: pd.DataFrame) -> List[Signal]:
        """Generate signals based on price thresholds."""
        signals = []

        today_data = data[data['date'] == current_date]

        if today_data.empty:
            return signals

        for _, row in today_data.iterrows():
            if row['outcome'] != 'Yes':
                continue

            market_id = row['market_id']
            token_id = row['token_id']
            price = row['price']
            position = self.get_position(market_id, 'Yes')

            # Buy signal
            if price < self.buy_threshold and position == 0:
                signals.append(Signal(
                    market_id=market_id,
                    token_id=token_id,
                    outcome='Yes',
                    signal_type=SignalType.BUY,
                    size=1.0,
                    price=price,
                    reason=f"Price {price:.3f} < buy threshold {self.buy_threshold}"
                ))

            # Sell signal
            elif price > self.sell_threshold and position > 0:
                signals.append(Signal(
                    market_id=market_id,
                    token_id=token_id,
                    outcome='Yes',
                    signal_type=SignalType.SELL,
                    size=position,
                    price=price,
                    reason=f"Price {price:.3f} > sell threshold {self.sell_threshold}"
                ))

        return signals
