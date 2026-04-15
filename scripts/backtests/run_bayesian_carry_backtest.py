#!/usr/bin/env python3
"""
Run Bayesian Carry Strategy Backtest with Three Hedging Variants

Implements a carry trade on short-dated prediction markets with:
- Full Bayesian posterior inference via NUTS (PyMC)
- Kelly criterion position sizing from posterior predictive uncertainty
- Three backtest variants: baseline, volume-hedged, cash-hedged

Strategy:
- Trade short-dated (≤30 days) No contracts
- BUY when posterior mean edge > min_edge (5%)
- Position sizing: fractional Kelly (25%) using posterior variance
  f* = kelly_fraction * (μ_edge / σ²_edge)
- Wide posterior (high uncertainty) → small position
- Narrow posterior (high confidence) → large position

Variants:
1. Baseline: Pure Kelly-sized carry (no hedging)
2. Volume Hedge: Exit when volume z-score > 2.0 (informed trading signal)
3. Cash Hedge: Dynamic cash reserve (20% base, +10% during 5% drawdowns)

Expected Outcomes:
- Baseline: Benchmark performance
- Volume Hedge: Better Sharpe if volume acceleration predicts adverse moves
- Cash Hedge: Lower max DD if cash buffering reduces risk without hurting returns
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from backtest import BacktestEngine, DataLoader
from backtest.strategies.bayesian_carry_strategy import BayesianCarryStrategy
import pandas as pd
import sqlite3
import numpy as np

DB_PATH = "data/polymarket.db"

# Get available date range for backtest
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
cursor.execute("SELECT MIN(date), MAX(date) FROM price_history")
min_date_str, max_date_str = cursor.fetchone()
conn.close()

# Convert to pandas timestamps
min_date = pd.Timestamp(min_date_str)
max_date = pd.Timestamp(max_date_str)

# Calculate total days in dataset
total_days = (max_date - min_date).days

# Use fixed EB cutoff date: learn from all data before 2024-06-01
eb_cutoff = pd.Timestamp('2024-06-01')
eb_learning_days = (eb_cutoff - min_date).days

# Set dates
eb_holdout_end = eb_cutoff.strftime('%Y-%m-%d')
backtest_start = (eb_cutoff + pd.DateOffset(days=1)).strftime('%Y-%m-%d')
backtest_end = max_date.strftime('%Y-%m-%d')

# Validate dates
if pd.Timestamp(backtest_start) >= max_date:
    raise ValueError(f"Backtest start ({backtest_start}) is after data end ({max_date_str}). "
                     f"Dataset may be too small. Total days: {total_days}")

# Base configuration (shared across variants)
base_config = {
    # Model parameters
    'mcmc_draws': 500,      # Reduced to avoid memory issues
    'mcmc_tune': 250,       # Reduced tuning steps
    'mcmc_chains': 2,
    'mcmc_cores': 2,        # Reduced to avoid memory issues
    'refit_days': 7,

    # Carry trade parameters
    'max_tte_days': 90,       # Medium-dated (was 30, too restrictive)
    'min_edge': 0.01,         # 1% minimum edge (was 0.05, too tight!)
    'kelly_fraction': 0.25,   # Quarter Kelly
    'max_position': 0.10,     # 10% portfolio cap

    # Empirical Bayes (CRITICAL for predictive power!)
    'use_eb_priors': True,
    'eb_holdout_end_date': eb_holdout_end,  # Use first year for EB learning
    'db_path': DB_PATH,

    # Initial capital
    'initial_capital': 10000.0,
}

# Variant configurations
variants = [
    ('baseline', {
        **base_config,
        'hedging': None,
        'verbose': True,
    }),
    ('volume_hedge', {
        **base_config,
        'hedging': 'volume',
        'vol_lookback_hours': 24,
        'vol_accel_threshold': 2.0,
        'verbose': True,
    }),
    ('cash_hedge', {
        **base_config,
        'hedging': 'cash',
        'min_cash_reserve': 0.20,
        'drawdown_threshold': 0.05,
        'drawdown_cash_add': 0.10,
        'max_cash_reserve': 0.50,
        'verbose': True,
    })
]

print(f"\n{'='*70}")
print("Bayesian Carry Strategy with EMPIRICAL BAYES Priors")
print(f"{'='*70}")
print(f"Strategy:              Kelly-sized carry on short-dated No contracts")
print(f"DB:                    {DB_PATH}")
print(f"Data Range:            {min_date_str} to {max_date_str} ({total_days} days)")
print(f"EB Learning Period:    {min_date_str} to {eb_holdout_end} ({eb_learning_days} days)")
print(f"Backtest Period:       {backtest_start} to {backtest_end} ({(max_date - pd.Timestamp(backtest_start)).days} days)")
print(f"Initial Capital:       ${base_config['initial_capital']:,.0f}")
print(f"Max TTE:               {base_config['max_tte_days']} days")
print(f"Kelly Fraction:        {base_config['kelly_fraction']*100:.0f}%")
print(f"Min Edge:              {base_config['min_edge']*100:.0f}%")
print(f"Max Position:          {base_config['max_position']*100:.0f}%")
print(f"Refit Interval:        {base_config['refit_days']} days")
print(f"MCMC:                  {base_config['mcmc_draws']} draws × {base_config['mcmc_chains']} chains")
print(f"\nKEY: EB priors learn from historical events, then predict new events")
print(f"     This creates REAL predictive power (not circular fitting)")
print(f"{'='*70}\n")

results = {}
data_loader = DataLoader(DB_PATH)

for variant_name, config in variants:
    print(f"\n{'='*70}")
    print(f"Running Variant: {variant_name.upper().replace('_', ' ')}")
    print(f"{'='*70}")

    if config['hedging'] == 'volume':
        print(f"Hedging:               Volume acceleration")
        print(f"  Lookback:            {config['vol_lookback_hours']} hours")
        print(f"  Exit threshold:      Z-score > {config['vol_accel_threshold']}")
        print(f"  Logic:               High volume = informed trading → exit")
    elif config['hedging'] == 'cash':
        print(f"Hedging:               Dynamic cash reserve")
        print(f"  Base reserve:        {config['min_cash_reserve']*100:.0f}%")
        print(f"  Drawdown trigger:    {config['drawdown_threshold']*100:.0f}%")
        print(f"  Reserve increase:    +{config['drawdown_cash_add']*100:.0f}%")
        print(f"  Max reserve:         {config['max_cash_reserve']*100:.0f}%")
    else:
        print(f"Hedging:               None (baseline)")

    print()

    # Create strategy
    strategy = BayesianCarryStrategy(config=config)

    # Create engine
    engine = BacktestEngine(
        strategy=strategy,
        data_loader=data_loader,
        initial_capital=base_config['initial_capital'],
        commission=0.0,
        max_position_size=base_config['max_position'],
        max_positions=None,
        verbose=True
    )

    # Run backtest (on out-of-sample period after EB learning)
    print("Starting backtest on out-of-sample period...\n")
    variant_results = engine.run(
        start_date=backtest_start,
        end_date=backtest_end,
        use_timing_markets=True,
        min_volume=100
    )

    if not variant_results:
        print(f"\n❌ {variant_name} backtest failed")
        continue

    results[variant_name] = variant_results

    # Print results
    print(f"\n{'='*70}")
    print(f"{variant_name.upper().replace('_', ' ')} RESULTS")
    print(f"{'='*70}\n")
    engine.print_results()

    # Save trades
    os.makedirs('backtest_results', exist_ok=True)
    if not variant_results['trades'].empty:
        out_file = f'backtest_results/bayesian_carry_{variant_name}_trades.csv'
        variant_results['trades'].to_csv(out_file, index=False)
        print(f"\n✓ Trades saved to: {out_file}")
    else:
        print(f"\n⚠ No trades executed for {variant_name}")

# Print comparison table
if len(results) > 0:
    print(f"\n\n{'='*70}")
    print("VARIANT COMPARISON")
    print(f"{'='*70}\n")

    # Build comparison DataFrame
    comparison_data = []
    for variant_name, result in results.items():
        # Calculate average Kelly size
        if not result['trades'].empty and 'metadata' in result['trades'].columns:
            try:
                avg_kelly = result['trades']['metadata'].apply(
                    lambda x: x.get('kelly_size', 0) if isinstance(x, dict) else 0
                ).mean()
            except:
                avg_kelly = 0
        else:
            avg_kelly = 0

        comparison_data.append({
            'Variant': variant_name.replace('_', ' ').title(),
            'Total Return (%)': f"{result['total_return']:.2f}",
            'CAGR (%)': f"{result['cagr']:.2f}",
            'Sharpe Ratio': f"{result['sharpe_ratio']:.2f}",
            'Sortino Ratio': f"{result['sortino_ratio']:.2f}",
            'Max Drawdown (%)': f"{result['max_drawdown']:.2f}",
            'Win Rate (%)': f"{result['win_rate']:.2f}",
            'Profit Factor': f"{result['profit_factor']:.2f}",
            'Total Trades': result['num_trades'],
            'Avg Kelly Size ($)': f"{avg_kelly:.2f}"
        })

    comparison_df = pd.DataFrame(comparison_data)

    # Print formatted table
    print(comparison_df.to_string(index=False))

    # Save comparison
    comparison_df.to_csv('backtest_results/bayesian_carry_comparison.csv', index=False)
    print(f"\n✓ Comparison saved to: backtest_results/bayesian_carry_comparison.csv")

    print(f"\n{'='*70}")
    print("INTERPRETATION GUIDE")
    print(f"{'='*70}")
    print("\nKey Questions:")
    print("  - Vol Hedge Sharpe > Baseline? → Volume acceleration is a real signal")
    print("  - Cash Hedge Max DD < Baseline with similar Sharpe? → Cash buffering is free")
    print("  - Avg Kelly Size shrinks in hedge variants? → Sizing responds correctly")
    print("\nExpected Patterns:")
    print("  - Baseline: Highest exposure, highest variance")
    print("  - Volume Hedge: Lower win rate, better Sharpe (avoid informed trades)")
    print("  - Cash Hedge: Lower max DD, smoother equity curve")

    print(f"\n{'='*70}")
    print("KELLY CRITERION INSIGHTS")
    print(f"{'='*70}")
    print("\nPosition Sizing Formula:")
    print("  f* = kelly_fraction * (μ_edge / σ²_edge)")
    print("\nKey Properties:")
    print("  - Wide posterior (high σ²) → small position (uncertain)")
    print("  - Narrow posterior (low σ²) → large position (confident)")
    print("  - Negative edge → no position")
    print("  - Hard cap at max_position (10% of portfolio)")
    print("\nBayesian Advantage:")
    print("  - Posterior variance naturally quantifies model uncertainty")
    print("  - Kelly sizing automatically adjusts for confidence")
    print("  - No ad-hoc position sizing rules needed")
    print(f"{'='*70}\n")

else:
    print("\n❌ No successful backtests to compare")

print("\nBacktest complete!")
