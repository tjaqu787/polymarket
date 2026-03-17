"""
Performance metrics calculation.
"""

import pandas as pd
import numpy as np
from typing import Optional


class PerformanceMetrics:
    """Calculate portfolio performance metrics."""

    def __init__(self, history_df: pd.DataFrame, trades_df: pd.DataFrame):
        """
        Initialize performance calculator.

        Args:
            history_df: Portfolio history DataFrame
            trades_df: Trades DataFrame
        """
        self.history = history_df.copy()
        self.trades = trades_df.copy()

        if not self.history.empty:
            self.history = self.history.sort_values('date')
            self.history['returns'] = self.history['total_value'].pct_change()

    def total_return(self) -> float:
        """Calculate total return percentage."""
        if self.history.empty or len(self.history) < 2:
            return 0.0

        initial = self.history.iloc[0]['total_value']
        final = self.history.iloc[-1]['total_value']

        if initial == 0:
            return 0.0

        return ((final - initial) / initial) * 100

    def cagr(self) -> float:
        """Calculate Compound Annual Growth Rate."""
        if self.history.empty or len(self.history) < 2:
            return 0.0

        initial = self.history.iloc[0]['total_value']
        final = self.history.iloc[-1]['total_value']
        start_date = self.history.iloc[0]['date']
        end_date = self.history.iloc[-1]['date']

        years = (end_date - start_date).days / 365.25

        if years <= 0 or initial <= 0:
            return 0.0

        return ((final / initial) ** (1 / years) - 1) * 100

    def daily_returns(self) -> pd.Series:
        """Get daily returns series."""
        if self.history.empty:
            return pd.Series()

        return self.history['returns'].fillna(0)

    def volatility(self, annualize: bool = True) -> float:
        """
        Calculate returns volatility (standard deviation).

        Args:
            annualize: If True, annualize the volatility

        Returns:
            Volatility (percentage)
        """
        returns = self.daily_returns()

        if len(returns) < 2:
            return 0.0

        vol = returns.std()

        if annualize:
            vol *= np.sqrt(252)  # Assume 252 trading days per year

        return vol * 100

    def sharpe_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sharpe ratio.

        Args:
            risk_free_rate: Annual risk-free rate (as percentage)

        Returns:
            Sharpe ratio
        """
        returns = self.daily_returns()

        if len(returns) < 2:
            return 0.0

        # Convert annual risk-free rate to daily
        daily_rf = (1 + risk_free_rate / 100) ** (1 / 252) - 1

        excess_returns = returns - daily_rf
        mean_excess = excess_returns.mean()
        std_excess = excess_returns.std()

        if std_excess == 0:
            return 0.0

        # Annualize
        sharpe = (mean_excess / std_excess) * np.sqrt(252)

        return sharpe

    def sortino_ratio(self, risk_free_rate: float = 0.0) -> float:
        """
        Calculate Sortino ratio (uses downside deviation).

        Args:
            risk_free_rate: Annual risk-free rate (as percentage)

        Returns:
            Sortino ratio
        """
        returns = self.daily_returns()

        if len(returns) < 2:
            return 0.0

        # Convert annual risk-free rate to daily
        daily_rf = (1 + risk_free_rate / 100) ** (1 / 252) - 1

        excess_returns = returns - daily_rf
        mean_excess = excess_returns.mean()

        # Calculate downside deviation (only negative returns)
        downside_returns = excess_returns[excess_returns < 0]

        if len(downside_returns) == 0:
            return np.inf if mean_excess > 0 else 0.0

        downside_std = downside_returns.std()

        if downside_std == 0:
            return 0.0

        # Annualize
        sortino = (mean_excess / downside_std) * np.sqrt(252)

        return sortino

    def max_drawdown(self) -> float:
        """
        Calculate maximum drawdown percentage.

        Returns:
            Max drawdown (as positive percentage)
        """
        if self.history.empty:
            return 0.0

        values = self.history['total_value'].values
        cummax = np.maximum.accumulate(values)
        drawdown = (values - cummax) / cummax * 100

        return abs(drawdown.min())

    def max_drawdown_duration(self) -> int:
        """
        Calculate maximum drawdown duration in days.

        Returns:
            Duration in days
        """
        if self.history.empty:
            return 0

        values = self.history['total_value'].values
        cummax = np.maximum.accumulate(values)
        underwater = values < cummax

        if not underwater.any():
            return 0

        # Find longest consecutive underwater period
        max_duration = 0
        current_duration = 0

        for is_underwater in underwater:
            if is_underwater:
                current_duration += 1
                max_duration = max(max_duration, current_duration)
            else:
                current_duration = 0

        return max_duration

    def win_rate(self) -> float:
        """Calculate win rate percentage."""
        if self.trades.empty:
            return 0.0

        winning_trades = (self.trades['pnl'] > 0).sum()
        total_trades = len(self.trades)

        if total_trades == 0:
            return 0.0

        return (winning_trades / total_trades) * 100

    def num_winning_trades(self) -> int:
        """Number of winning trades."""
        if self.trades.empty:
            return 0
        return (self.trades['pnl'] > 0).sum()

    def num_losing_trades(self) -> int:
        """Number of losing trades."""
        if self.trades.empty:
            return 0
        return (self.trades['pnl'] < 0).sum()

    def profit_factor(self) -> float:
        """
        Calculate profit factor (gross profit / gross loss).

        Returns:
            Profit factor
        """
        if self.trades.empty:
            return 0.0

        gross_profit = self.trades[self.trades['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(self.trades[self.trades['pnl'] < 0]['pnl'].sum())

        if gross_loss == 0:
            return np.inf if gross_profit > 0 else 0.0

        return gross_profit / gross_loss

    def avg_win(self) -> float:
        """Average winning trade P&L."""
        if self.trades.empty:
            return 0.0

        winning_trades = self.trades[self.trades['pnl'] > 0]['pnl']

        if len(winning_trades) == 0:
            return 0.0

        return winning_trades.mean()

    def avg_loss(self) -> float:
        """Average losing trade P&L."""
        if self.trades.empty:
            return 0.0

        losing_trades = self.trades[self.trades['pnl'] < 0]['pnl']

        if len(losing_trades) == 0:
            return 0.0

        return losing_trades.mean()

    def avg_trade(self) -> float:
        """Average trade P&L."""
        if self.trades.empty:
            return 0.0

        return self.trades['pnl'].mean()

    def expectancy(self) -> float:
        """
        Calculate trade expectancy.

        Returns:
            Expected P&L per trade
        """
        if self.trades.empty:
            return 0.0

        win_rate = self.win_rate() / 100
        avg_win = self.avg_win()
        avg_loss = abs(self.avg_loss())

        return (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    def calmar_ratio(self) -> float:
        """
        Calculate Calmar ratio (CAGR / Max Drawdown).

        Returns:
            Calmar ratio
        """
        cagr = self.cagr()
        max_dd = self.max_drawdown()

        if max_dd == 0:
            return np.inf if cagr > 0 else 0.0

        return cagr / max_dd

    def get_summary(self) -> dict:
        """
        Get comprehensive performance summary.

        Returns:
            Dictionary with all metrics
        """
        return {
            'total_return': self.total_return(),
            'cagr': self.cagr(),
            'volatility': self.volatility(),
            'sharpe_ratio': self.sharpe_ratio(),
            'sortino_ratio': self.sortino_ratio(),
            'max_drawdown': self.max_drawdown(),
            'max_drawdown_duration': self.max_drawdown_duration(),
            'win_rate': self.win_rate(),
            'profit_factor': self.profit_factor(),
            'avg_win': self.avg_win(),
            'avg_loss': self.avg_loss(),
            'avg_trade': self.avg_trade(),
            'expectancy': self.expectancy(),
            'calmar_ratio': self.calmar_ratio(),
            'num_trades': len(self.trades),
            'num_winning_trades': self.num_winning_trades(),
            'num_losing_trades': self.num_losing_trades(),
        }
