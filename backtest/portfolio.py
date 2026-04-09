"""
Portfolio management and P&L tracking.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Position:
    """Represents a market position."""
    market_id: str
    token_id: str
    outcome: str
    size: float
    entry_price: float
    entry_date: pd.Timestamp
    current_price: float = 0.0
    unrealized_pnl: float = 0.0
    metadata: Dict = field(default_factory=dict)

    def update_price(self, price: float):
        """Update current price and unrealized P&L."""
        self.current_price = price
        self.unrealized_pnl = (price - self.entry_price) * self.size


@dataclass
class Trade:
    """Represents a completed trade."""
    trade_id: int
    market_id: str
    token_id: str
    outcome: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    size: float
    pnl: float
    return_pct: float
    holding_period_days: int
    reason: str = ""
    metadata: Dict = field(default_factory=dict)


@dataclass
class PortfolioSnapshot:
    """Portfolio state at a point in time."""
    date: pd.Timestamp
    cash: float
    positions_value: float
    total_value: float
    num_positions: int
    unrealized_pnl: float


class Portfolio:
    """
    Manages portfolio positions, cash, and P&L tracking.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        commission: float = 0.0,
        max_position_size: float = 0.1,
        max_positions: Optional[int] = None
    ):
        """
        Initialize portfolio.

        Args:
            initial_capital: Starting capital
            commission: Commission per trade (fraction, e.g., 0.01 for 1%)
            max_position_size: Max position size as fraction of portfolio value
            max_positions: Maximum number of concurrent positions (None = unlimited)
        """
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.commission = commission
        self.max_position_size = max_position_size
        self.max_positions = max_positions

        self.positions: Dict[tuple, Position] = {}  # (market_id, outcome) -> Position
        self.trades: List[Trade] = []
        self.history: List[PortfolioSnapshot] = []

        self.trade_counter = 0

    @property
    def total_value(self) -> float:
        """Total portfolio value (cash + positions)."""
        positions_value = sum(p.current_price * p.size for p in self.positions.values())
        return self.cash + positions_value

    @property
    def positions_value(self) -> float:
        """Total value of all positions."""
        return sum(p.current_price * p.size for p in self.positions.values())

    @property
    def unrealized_pnl(self) -> float:
        """Total unrealized P&L."""
        return sum(p.unrealized_pnl for p in self.positions.values())

    @property
    def num_positions(self) -> int:
        """Number of open positions."""
        return len(self.positions)

    def can_open_position(self) -> bool:
        """Check if we can open a new position."""
        if self.max_positions is None:
            return True
        return self.num_positions < self.max_positions

    def calculate_position_size(self, price: float) -> float:
        """
        Calculate position size based on portfolio constraints.

        Args:
            price: Entry price

        Returns:
            Position size (number of contracts)
        """
        if price <= 0:
            return 0

        max_value = self.total_value * self.max_position_size
        size = max_value / price

        # Ensure we have enough cash
        required_cash = size * price * (1 + self.commission)
        if required_cash > self.cash:
            size = self.cash / (price * (1 + self.commission))

        return max(0, size)

    def open_position(
        self,
        market_id: str,
        token_id: str,
        outcome: str,
        price: float,
        size: float,
        date: pd.Timestamp,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        Open a new position.

        Args:
            market_id: Market ID
            token_id: Token ID
            outcome: Outcome (Yes/No)
            price: Entry price
            size: Position size
            date: Entry date
            metadata: Optional metadata (e.g., for tracking calendar spread pairs)

        Returns:
            True if position opened successfully
        """
        key = (market_id, outcome)

        # Check if position already exists
        if key in self.positions:
            return False

        # Check position limit
        if not self.can_open_position():
            return False

        # Calculate cost
        cost = price * size * (1 + self.commission)

        # Check cash
        if cost > self.cash:
            return False

        # Open position
        self.cash -= cost
        self.positions[key] = Position(
            market_id=market_id,
            token_id=token_id,
            outcome=outcome,
            size=size,
            entry_price=price,
            entry_date=date,
            current_price=price,
            metadata=metadata or {}
        )

        return True

    def close_position(
        self,
        market_id: str,
        outcome: str,
        price: float,
        date: pd.Timestamp,
        reason: str = ""
    ) -> Optional[Trade]:
        """
        Close an existing position.

        Args:
            market_id: Market ID
            outcome: Outcome
            price: Exit price
            date: Exit date
            reason: Reason for closing

        Returns:
            Trade object if closed successfully, None otherwise
        """
        key = (market_id, outcome)

        if key not in self.positions:
            return None

        position = self.positions[key]

        # Calculate proceeds
        proceeds = price * position.size * (1 - self.commission)
        self.cash += proceeds

        # Calculate P&L
        cost = position.entry_price * position.size
        pnl = proceeds - cost - (cost * self.commission)  # Include entry commission
        return_pct = (pnl / cost) * 100 if cost > 0 else 0

        # Create trade record
        holding_days = (date - position.entry_date).days
        trade = Trade(
            trade_id=self.trade_counter,
            market_id=market_id,
            token_id=position.token_id,
            outcome=outcome,
            entry_date=position.entry_date,
            exit_date=date,
            entry_price=position.entry_price,
            exit_price=price,
            size=position.size,
            pnl=pnl,
            return_pct=return_pct,
            holding_period_days=holding_days,
            reason=reason,
            metadata=position.metadata
        )

        self.trades.append(trade)
        self.trade_counter += 1

        # Remove position
        del self.positions[key]

        return trade

    def update_prices(self, prices: Dict[tuple, float]):
        """
        Update current prices for all positions.

        Args:
            prices: Dict mapping (market_id, outcome) to current price
        """
        for key, position in self.positions.items():
            if key in prices:
                position.update_price(prices[key])

    def snapshot(self, date: pd.Timestamp) -> PortfolioSnapshot:
        """
        Create a snapshot of current portfolio state.

        Args:
            date: Current date

        Returns:
            PortfolioSnapshot
        """
        snapshot = PortfolioSnapshot(
            date=date,
            cash=self.cash,
            positions_value=self.positions_value,
            total_value=self.total_value,
            num_positions=self.num_positions,
            unrealized_pnl=self.unrealized_pnl
        )
        self.history.append(snapshot)
        return snapshot

    def get_position(self, market_id: str, outcome: str) -> Optional[Position]:
        """Get position for a market."""
        key = (market_id, outcome)
        return self.positions.get(key)

    def has_position(self, market_id: str, outcome: str) -> bool:
        """Check if position exists."""
        return (market_id, outcome) in self.positions

    def get_trades_df(self) -> pd.DataFrame:
        """Get trades as DataFrame with metadata columns."""
        if not self.trades:
            return pd.DataFrame()

        records = []
        for t in self.trades:
            record = {
                'trade_id': t.trade_id,
                'market_id': t.market_id,
                'token_id': t.token_id,
                'outcome': t.outcome,
                'entry_date': t.entry_date,
                'exit_date': t.exit_date,
                'entry_price': t.entry_price,
                'exit_price': t.exit_price,
                'size': t.size,
                'pnl': t.pnl,
                'return_pct': t.return_pct,
                'holding_period_days': t.holding_period_days,
                'reason': t.reason
            }
            # Flatten metadata into columns
            if t.metadata:
                # Add common spread trading metadata
                record['trade_type'] = t.metadata.get('trade_type', '')
                record['pair_id'] = t.metadata.get('pair_id', '')
                record['spread_level'] = t.metadata.get('spread_level', np.nan)
                record['spread_velocity'] = t.metadata.get('spread_velocity', np.nan)
                record['vol_zscore'] = t.metadata.get('vol_zscore', np.nan)
                record['regime_factor'] = t.metadata.get('regime_factor', np.nan)
                record['volume'] = t.metadata.get('volume', np.nan)
                record['leg'] = t.metadata.get('leg', '')
                record['entry_spread'] = t.metadata.get('entry_spread', np.nan)
                record['exit_spread'] = t.metadata.get('exit_spread', np.nan)
            else:
                # Fill with NaN if no metadata
                record.update({
                    'trade_type': '',
                    'pair_id': '',
                    'spread_level': np.nan,
                    'spread_velocity': np.nan,
                    'vol_zscore': np.nan,
                    'regime_factor': np.nan,
                    'volume': np.nan,
                    'leg': '',
                    'entry_spread': np.nan,
                    'exit_spread': np.nan
                })
            records.append(record)

        return pd.DataFrame(records)

    def get_history_df(self) -> pd.DataFrame:
        """Get portfolio history as DataFrame."""
        if not self.history:
            return pd.DataFrame()

        return pd.DataFrame([
            {
                'date': s.date,
                'cash': s.cash,
                'positions_value': s.positions_value,
                'total_value': s.total_value,
                'num_positions': s.num_positions,
                'unrealized_pnl': s.unrealized_pnl
            }
            for s in self.history
        ])
