#!/usr/bin/env python3
"""
Prior Sensitivity Analysis for Bayesian Kelly Criterion

Runs the spread dynamics backtest with different prior_edge_std values
to analyze sensitivity to prior assumptions.
"""

import sys
import os
import numpy as np
import pandas as pd
from pathlib import Path
import importlib.util
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

sys.path.insert(0, os.path.dirname(__file__))

from backtest import BacktestEngine, DataLoader

# Import strategy
spec = importlib.util.spec_from_file_location(
    "spread_dynamics_strategy",
    "backtest/strategies/spread_dynamics_strategy.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SpreadDynamicsStrategy = module.SpreadDynamicsStrategy

DB_PATH = "data/polymarket.db"

# Define prior_std values to test
# Uniformly sample 30 values between 0.01 and 0.30
N_SAMPLES = 30
PRIOR_STD_VALUES = np.linspace(0.01, 0.30, N_SAMPLES).tolist()

def run_backtest_with_prior(prior_std, verbose=False):
    """Run backtest with specified prior_std value."""

    # Add some error handling and resource management
    try:
        strategy = SpreadDynamicsStrategy(config={
            "db_path": DB_PATH,
            "lookback_days": 7,
            "vol_lookback_days": 3,
            "vol_spike_threshold": 2.0,
            "vol_drought_threshold": -0.5,
            "min_spread_change": 0.05,
            "min_spread_level": 0.02,
            "min_tte_days": 14,
            "min_volume": 300,
            "kelly_fraction": 0.25,
            "max_position": 0.10,
            "max_event_exposure": 0.20,
            "min_pair_correlation": 0.60,
            # Bayesian Kelly parameters - varying prior_edge_std
            "use_bayesian_kelly": True,
            "prior_edge_mean": 0.0,
            "prior_edge_std": prior_std,  # THIS IS WHAT WE'RE VARYING
            "obs_std": 0.02,
        })

        data_loader = DataLoader(DB_PATH)

        engine = BacktestEngine(
            strategy=strategy,
            data_loader=data_loader,
            initial_capital=10_000.0,
            commission=0.0,
            max_position_size=0.1,
            max_positions=None,
            verbose=False,  # Suppress output for batch processing
        )

        results = engine.run(
            start_date="2024-01-01",
            end_date="2026-03-16",
            use_timing_markets=True,
            min_volume=300,
        )

        if results is None or not results:
            return None

        # Extract key metrics
        trades = results.get('trades', [])
        if trades is None or (isinstance(trades, list) and len(trades) == 0):
            return None

        trades_df = pd.DataFrame(trades)
        if trades_df.empty:
            return None
        total_pnl = trades_df['pnl'].sum()
        n_trades = len(trades_df)
        win_rate = (trades_df['pnl'] > 0).mean() * 100

        # Get posterior stats
        posterior_stats = strategy.get_posterior_stats()
        avg_posterior_std = posterior_stats['posterior_std'].mean() if not posterior_stats.empty else np.nan
        avg_uncertainty_reduction = posterior_stats['uncertainty_reduction'].mean() * 100 if not posterior_stats.empty else np.nan

        return {
            'prior_std': prior_std,
            'total_pnl': total_pnl,
            'n_trades': n_trades,
            'win_rate': win_rate,
            'avg_posterior_std': avg_posterior_std,
            'avg_uncertainty_reduction': avg_uncertainty_reduction,
            'mean_return': trades_df['return_pct'].mean(),
            'std_return': trades_df['return_pct'].std(),
        }

    except Exception as e:
        # Log error and return None
        print(f"\n⚠️  Error in backtest with prior_std={prior_std:.3f}: {str(e)[:100]}")
        return None


def main():
    print("="*70)
    print("BAYESIAN KELLY PRIOR SENSITIVITY ANALYSIS (MULTITHREADED)")
    print("="*70)
    print(f"\nTesting {len(PRIOR_STD_VALUES)} different prior_edge_std values...")
    print(f"Prior std range: [{min(PRIOR_STD_VALUES):.3f}, {max(PRIOR_STD_VALUES):.3f}]")

    # Use fewer workers to avoid memory issues and database locking
    # SQLite can have issues with many concurrent readers
    n_workers = min(4, max(1, cpu_count() - 2))
    print(f"Using {n_workers} parallel workers")
    print("\nRunning backtests in parallel...\n")

    results = []

    # Run backtests in parallel
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all jobs
        future_to_prior = {
            executor.submit(run_backtest_with_prior, prior_std): prior_std
            for prior_std in PRIOR_STD_VALUES
        }

        # Process results as they complete
        completed = 0
        for future in as_completed(future_to_prior):
            prior_std = future_to_prior[future]
            completed += 1

            try:
                result = future.result()
                if result:
                    results.append(result)
                    print(f"[{completed}/{len(PRIOR_STD_VALUES)}] ✓ prior_std={prior_std:.3f} → PnL: ${result['total_pnl']:.4f}, Trades: {result['n_trades']}")
                else:
                    print(f"[{completed}/{len(PRIOR_STD_VALUES)}] ✗ prior_std={prior_std:.3f} → Failed")
            except Exception as e:
                print(f"[{completed}/{len(PRIOR_STD_VALUES)}] ✗ prior_std={prior_std:.3f} → Error: {e}")

    if not results:
        print("\n❌ No successful backtests. Cannot generate analysis.")
        return

    # Convert to DataFrame and sort by prior_std
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('prior_std').reset_index(drop=True)

    # Save results
    output_dir = Path("backtest_results")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "prior_sensitivity_results.csv"
    results_df.to_csv(output_file, index=False)

    print("\n" + "="*70)
    print("RESULTS SUMMARY")
    print("="*70)
    print(f"\nResults saved to: {output_file}")
    print(f"\nTotal Runs: {len(results_df)}")
    print(f"\nPrior Std vs Total PnL:")
    print(results_df[['prior_std', 'total_pnl', 'n_trades', 'win_rate']].to_string(index=False))

    print(f"\n" + "="*70)
    print("KEY FINDINGS")
    print("="*70)

    best_idx = results_df['total_pnl'].idxmax()
    worst_idx = results_df['total_pnl'].idxmin()

    print(f"\n📈 Best Prior Std:  {results_df.loc[best_idx, 'prior_std']:.3f}")
    print(f"   → Total PnL: ${results_df.loc[best_idx, 'total_pnl']:.4f}")
    print(f"   → Trades: {results_df.loc[best_idx, 'n_trades']}")

    print(f"\n📉 Worst Prior Std: {results_df.loc[worst_idx, 'prior_std']:.3f}")
    print(f"   → Total PnL: ${results_df.loc[worst_idx, 'total_pnl']:.4f}")
    print(f"   → Trades: {results_df.loc[worst_idx, 'n_trades']}")

    pnl_range = results_df['total_pnl'].max() - results_df['total_pnl'].min()
    print(f"\n📊 PnL Range: ${pnl_range:.4f}")
    print(f"   (Max - Min across all priors)")

    print(f"\n💡 Interpretation:")
    if pnl_range < 5:
        print(f"   Result is ROBUST to prior choice (range < $5)")
        print(f"   → The 'financially irrelevant' conclusion holds across all priors")
    elif pnl_range < 20:
        print(f"   Result shows MODERATE sensitivity to prior (range < $20)")
        print(f"   → Still financially irrelevant regardless of prior")
    else:
        print(f"   Result shows HIGH sensitivity to prior (range > $20)")
        print(f"   → Prior choice significantly affects performance")

    print("="*70)


if __name__ == "__main__":
    main()
