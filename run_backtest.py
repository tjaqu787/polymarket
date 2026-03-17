#!/usr/bin/env python3
"""
Polymarket Backtest Runner

Example usage:
    python run_backtest.py --strategy mean_reversion --start 2024-01-01 --end 2024-12-31
    python run_backtest.py --strategy momentum --capital 20000 --commission 0.01
    python run_backtest.py --strategy term_structure --min-volume 1000
"""

import argparse
import sys
import os
from datetime import datetime

# Add current directory to path
sys.path.insert(0, os.path.dirname(__file__))

from backtest import BacktestEngine, DataLoader
from backtest.strategy import BuyAndHoldStrategy, ThresholdStrategy
from backtest.strategies import TermStructureStrategy, MeanReversionStrategy, MomentumStrategy
from backtest.visualization import plot_backtest_results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run Polymarket backtest')

    # Strategy selection
    parser.add_argument(
        '--strategy',
        type=str,
        default='mean_reversion',
        choices=['buy_and_hold', 'threshold', 'mean_reversion', 'momentum', 'term_structure'],
        help='Trading strategy to test'
    )

    # Date range
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')

    # Portfolio parameters
    parser.add_argument('--capital', type=float, default=10000, help='Initial capital')
    parser.add_argument('--commission', type=float, default=0.0, help='Commission rate (e.g., 0.01 for 1%%)')
    parser.add_argument('--max-position-size', type=float, default=0.1, help='Max position size as fraction of portfolio')
    parser.add_argument('--max-positions', type=int, help='Max concurrent positions (default: unlimited)')

    # Data filters
    parser.add_argument('--min-volume', type=float, default=100, help='Minimum market volume')
    parser.add_argument('--market-ids', nargs='+', help='Specific market IDs to test')
    parser.add_argument('--event-ids', nargs='+', help='Specific event IDs to test')

    # Strategy-specific parameters
    parser.add_argument('--param', action='append', help='Strategy parameter (format: key=value)')

    # Output options
    parser.add_argument('--output-dir', type=str, default='backtest_results', help='Output directory for results')
    parser.add_argument('--plot', action='store_true', help='Generate plots')
    parser.add_argument('--save-trades', action='store_true', help='Save trades to CSV')

    # Database
    parser.add_argument('--db-path', type=str, default='data/polymarket.db', help='Path to database')

    return parser.parse_args()


def parse_strategy_params(param_list):
    """Parse strategy parameters from command line."""
    params = {}

    if not param_list:
        return params

    for param in param_list:
        if '=' not in param:
            print(f"Warning: Invalid parameter format '{param}', expected key=value")
            continue

        key, value = param.split('=', 1)

        # Try to convert to appropriate type
        try:
            # Try int
            params[key] = int(value)
        except ValueError:
            try:
                # Try float
                params[key] = float(value)
            except ValueError:
                # Keep as string
                params[key] = value

    return params


def create_strategy(strategy_name: str, params: dict):
    """Create strategy instance."""
    if strategy_name == 'buy_and_hold':
        return BuyAndHoldStrategy(params)

    elif strategy_name == 'threshold':
        return ThresholdStrategy(params)

    elif strategy_name == 'mean_reversion':
        return MeanReversionStrategy(params)

    elif strategy_name == 'momentum':
        return MomentumStrategy(params)

    elif strategy_name == 'term_structure':
        return TermStructureStrategy(params)

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")


def main():
    """Run backtest."""
    args = parse_args()

    # Parse strategy parameters
    strategy_params = parse_strategy_params(args.param)

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Initialize data loader
    data_loader = DataLoader(args.db_path)

    # Create strategy
    strategy = create_strategy(args.strategy, strategy_params)

    print(f"\n{'='*60}")
    print(f"Polymarket Backtest")
    print(f"{'='*60}")
    print(f"Strategy: {strategy.name}")
    print(f"Period: {args.start or 'earliest'} to {args.end or 'latest'}")
    print(f"Initial Capital: ${args.capital:,.2f}")
    print(f"Commission: {args.commission*100:.2f}%")
    print(f"{'='*60}\n")

    # Create backtest engine
    engine = BacktestEngine(
        strategy=strategy,
        data_loader=data_loader,
        initial_capital=args.capital,
        commission=args.commission,
        max_position_size=args.max_position_size,
        max_positions=args.max_positions,
        verbose=True
    )

    # Run backtest
    results = engine.run(
        start_date=args.start,
        end_date=args.end,
        market_ids=args.market_ids,
        event_ids=args.event_ids,
        use_timing_markets=True,
        min_volume=args.min_volume
    )

    if not results:
        print("\nBacktest failed. No results generated.")
        return

    # Print results
    engine.print_results()

    # Save trades
    if args.save_trades and not results['trades'].empty:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        trades_file = os.path.join(args.output_dir, f'trades_{strategy.name}_{timestamp}.csv')
        results['trades'].to_csv(trades_file, index=False)
        print(f"Trades saved to: {trades_file}")

    # Generate plots
    if args.plot:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        plot_file = os.path.join(args.output_dir, f'backtest_{strategy.name}_{timestamp}.html')

        plot_backtest_results(
            results=results,
            output_file=plot_file
        )

        print(f"Plot saved to: {plot_file}")


if __name__ == '__main__':
    main()
