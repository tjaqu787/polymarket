"""
Main backtesting engine.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Callable
from datetime import datetime
import logging

from .strategy import Strategy, Signal, SignalType
from .portfolio import Portfolio
from .data_loader import DataLoader
from .performance import PerformanceMetrics


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Main backtesting engine that orchestrates strategy execution
    over historical data.
    """

    def __init__(
        self,
        strategy: Strategy,
        data_loader: DataLoader,
        initial_capital: float = 10000.0,
        commission: float = 0.0,
        max_position_size: float = 0.1,
        max_positions: Optional[int] = None,
        verbose: bool = True
    ):
        """
        Initialize backtest engine.

        Args:
            strategy: Trading strategy to test
            data_loader: Data loader instance
            initial_capital: Starting capital
            commission: Commission per trade (fraction)
            max_position_size: Max position size as fraction of portfolio
            max_positions: Max concurrent positions
            verbose: Print progress messages
        """
        self.strategy = strategy
        self.data_loader = data_loader
        self.initial_capital = initial_capital
        self.commission = commission
        self.max_position_size = max_position_size
        self.max_positions = max_positions
        self.verbose = verbose

        self.portfolio = None
        self.data = None
        self.results = None

    def run(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        market_ids: Optional[List[str]] = None,
        event_ids: Optional[List[str]] = None,
        use_timing_markets: bool = True,
        min_volume: float = 100
    ) -> Dict:
        """
        Run backtest over specified date range and markets.

        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            market_ids: Specific market IDs to test
            event_ids: Specific event IDs to test
            use_timing_markets: Use timing markets view
            min_volume: Minimum market volume

        Returns:
            Dictionary with backtest results
        """
        logger.info(f"Starting backtest: {self.strategy.name}")
        logger.info(f"Period: {start_date} to {end_date}")
        logger.info(f"Initial capital: ${self.initial_capital:,.2f}")

        # Load data
        if use_timing_markets:
            self.data = self.data_loader.load_timing_markets(
                start_date=start_date,
                end_date=end_date,
                min_volume=min_volume
            )
        else:
            self.data = self.data_loader.load_market_data(
                market_ids=market_ids,
                event_ids=event_ids,
                start_date=start_date,
                end_date=end_date,
                min_volume=min_volume
            )

        if self.data.empty:
            logger.error("No data loaded. Check filters.")
            return {}

        logger.info(f"Loaded {len(self.data)} price records")
        logger.info(f"Markets: {self.data['market_id'].nunique()}")
        logger.info(f"Date range: {self.data['date'].min()} to {self.data['date'].max()}")

        # Initialize portfolio
        self.portfolio = Portfolio(
            initial_capital=self.initial_capital,
            commission=self.commission,
            max_position_size=self.max_position_size,
            max_positions=self.max_positions
        )

        # Reset strategy and inject portfolio reference
        self.strategy.reset()
        self.strategy.set_portfolio(self.portfolio)

        # Get unique dates
        dates = sorted(self.data['date'].unique())
        logger.info(f"Simulating {len(dates)} trading days")

        # Run simulation
        for i, current_date in enumerate(dates):
            if self.verbose and i % 30 == 0:
                logger.info(f"Progress: {i}/{len(dates)} days ({i/len(dates)*100:.1f}%)")

            # Get historical data up to current date
            historical_data = self.data[self.data['date'] <= current_date].copy()

            # Check for market resolutions
            self._check_resolutions(current_date, historical_data)

            # Update position prices
            self._update_prices(current_date, historical_data)

            # Generate signals
            signals = self.strategy.on_data(current_date, historical_data)

            # Execute signals
            self._execute_signals(signals, current_date)

            # Take portfolio snapshot
            self.portfolio.snapshot(current_date)

        # Close any remaining positions
        self._close_remaining_positions(dates[-1])

        logger.info("Backtest complete")
        logger.info(f"Final portfolio value: ${self.portfolio.total_value:,.2f}")
        logger.info(f"Total trades: {len(self.portfolio.trades)}")

        # Calculate performance metrics
        self.results = self._calculate_results()

        return self.results

    def _check_resolutions(self, current_date: pd.Timestamp, data: pd.DataFrame):
        """Check for markets that have resolved and close positions."""
        for (market_id, outcome), position in list(self.portfolio.positions.items()):
            # Get market resolution date
            market_data = data[
                (data['market_id'] == market_id) &
                (data['outcome'] == outcome)
            ]

            if market_data.empty:
                continue

            resolution_date = market_data.iloc[-1]['resolution_date']

            # Check if market has resolved
            if current_date >= resolution_date:
                # Determine settlement price
                # In reality, this would be 1.0 for winning outcome, 0.0 for losing
                # For now, we'll use the last available price
                last_price = market_data.iloc[-1]['price']

                # Close position
                self.portfolio.close_position(
                    market_id=market_id,
                    outcome=outcome,
                    price=last_price,
                    date=current_date,
                    reason="Market resolved"
                )

    def _update_prices(self, current_date: pd.Timestamp, data: pd.DataFrame):
        """Update current prices for all positions."""
        prices = {}

        for (market_id, outcome), position in self.portfolio.positions.items():
            # Get latest price for this market
            market_data = data[
                (data['market_id'] == market_id) &
                (data['outcome'] == outcome) &
                (data['date'] == current_date)
            ]

            if not market_data.empty:
                prices[(market_id, outcome)] = market_data.iloc[-1]['price']

        self.portfolio.update_prices(prices)

    def _execute_signals(self, signals: List[Signal], current_date: pd.Timestamp):
        """Execute trading signals."""
        for signal in signals:
            if signal.signal_type == SignalType.BUY or signal.signal_type == SignalType.SHORT:
                # Calculate position size if not specified
                if signal.size <= 0:
                    signal.size = self.portfolio.calculate_position_size(signal.price)

                # For SHORT, use negative size to indicate short position
                if signal.signal_type == SignalType.SHORT:
                    signal.size = -abs(signal.size)

                # Open position (long or short)
                success = self.portfolio.open_position(
                    market_id=signal.market_id,
                    token_id=signal.token_id,
                    outcome=signal.outcome,
                    price=signal.price,
                    size=signal.size,
                    date=current_date
                )

                if success and self.verbose:
                    position_type = "short" if signal.signal_type == SignalType.SHORT else "long"
                    logger.debug(f"Opened {position_type} position: {signal.market_id} @ {signal.price:.3f}")

            elif signal.signal_type in [SignalType.SELL, SignalType.COVER, SignalType.CLOSE]:
                # Close position (long or short)
                trade = self.portfolio.close_position(
                    market_id=signal.market_id,
                    outcome=signal.outcome,
                    price=signal.price,
                    date=current_date,
                    reason=signal.reason or "Signal"
                )

                if trade and self.verbose:
                    logger.debug(f"Closed position: {signal.market_id} @ {signal.price:.3f}, P&L: ${trade.pnl:.2f}")

    def _close_remaining_positions(self, final_date: pd.Timestamp):
        """Close all remaining positions at end of backtest."""
        for (market_id, outcome), position in list(self.portfolio.positions.items()):
            # Use last available price
            market_data = self.data[
                (self.data['market_id'] == market_id) &
                (self.data['outcome'] == outcome)
            ]

            if not market_data.empty:
                last_price = market_data.iloc[-1]['price']
                self.portfolio.close_position(
                    market_id=market_id,
                    outcome=outcome,
                    price=last_price,
                    date=final_date,
                    reason="End of backtest"
                )

    def _calculate_results(self) -> Dict:
        """Calculate backtest results and performance metrics."""
        trades_df = self.portfolio.get_trades_df()
        history_df = self.portfolio.get_history_df()

        if history_df.empty:
            return {
                'strategy_name': self.strategy.name,
                'initial_capital': self.initial_capital,
                'final_value': self.initial_capital,
                'total_return': 0.0,
                'num_trades': 0,
                'trades': trades_df,
                'history': history_df
            }

        # Calculate performance metrics
        metrics = PerformanceMetrics(history_df, trades_df)

        results = {
            'strategy_name': self.strategy.name,
            'initial_capital': self.initial_capital,
            'final_value': self.portfolio.total_value,
            'total_return': metrics.total_return(),
            'cagr': metrics.cagr(),
            'sharpe_ratio': metrics.sharpe_ratio(),
            'sortino_ratio': metrics.sortino_ratio(),
            'max_drawdown': metrics.max_drawdown(),
            'max_drawdown_duration': metrics.max_drawdown_duration(),
            'win_rate': metrics.win_rate(),
            'profit_factor': metrics.profit_factor(),
            'avg_win': metrics.avg_win(),
            'avg_loss': metrics.avg_loss(),
            'avg_trade': metrics.avg_trade(),
            'num_trades': len(trades_df),
            'num_winning_trades': metrics.num_winning_trades(),
            'num_losing_trades': metrics.num_losing_trades(),
            'trades': trades_df,
            'history': history_df,
            'daily_returns': metrics.daily_returns(),
        }

        return results

    def print_results(self):
        """Print backtest results summary."""
        if not self.results:
            logger.error("No results available. Run backtest first.")
            return

        print("\n" + "="*60)
        print(f"BACKTEST RESULTS: {self.results['strategy_name']}")
        print("="*60)

        print(f"\nPortfolio Performance:")
        print(f"  Initial Capital:        ${self.results['initial_capital']:>12,.2f}")
        print(f"  Final Value:            ${self.results['final_value']:>12,.2f}")
        print(f"  Total Return:           {self.results['total_return']:>12.2f}%")
        print(f"  CAGR:                   {self.results['cagr']:>12.2f}%")

        print(f"\nRisk Metrics:")
        print(f"  Sharpe Ratio:           {self.results['sharpe_ratio']:>12.2f}")
        print(f"  Sortino Ratio:          {self.results['sortino_ratio']:>12.2f}")
        print(f"  Max Drawdown:           {self.results['max_drawdown']:>12.2f}%")
        print(f"  Max DD Duration:        {self.results['max_drawdown_duration']:>12} days")

        print(f"\nTrade Statistics:")
        print(f"  Total Trades:           {self.results['num_trades']:>12}")
        print(f"  Winning Trades:         {self.results['num_winning_trades']:>12}")
        print(f"  Losing Trades:          {self.results['num_losing_trades']:>12}")
        print(f"  Win Rate:               {self.results['win_rate']:>12.2f}%")
        print(f"  Profit Factor:          {self.results['profit_factor']:>12.2f}")
        print(f"  Avg Win:                ${self.results['avg_win']:>12.2f}")
        print(f"  Avg Loss:               ${self.results['avg_loss']:>12.2f}")
        print(f"  Avg Trade:              ${self.results['avg_trade']:>12.2f}")

        print("\n" + "="*60 + "\n")
