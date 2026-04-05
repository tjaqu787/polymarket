#!/usr/bin/env python3
"""
Run backtest using the SpreadDynamicsStrategy.

This strategy trades anticipated changes in implied rate spreads between
adjacent contracts in term structures, using volume regime and spread
velocity as signals.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from backtest import BacktestEngine, DataLoader
from backtest.strategies.spread_dynamics_strategy import SpreadDynamicsStrategy

DB_PATH = "data/polymarket.db"

strategy = SpreadDynamicsStrategy(config={
    "db_path": DB_PATH,
    "lookback_days": 7,              # Window for measuring spread change
    "vol_lookback_days": 3,          # Window for volume regime classification
    "vol_spike_threshold": 2.0,      # Z-score for "news event" regime
    "vol_drought_threshold": -0.5,   # Z-score for "inactive" regime
    "min_spread_change": 0.05,       # Minimum |Δspread| to generate signal
    "min_spread_level": 0.02,        # Ignore pairs with near-zero spread
    "min_tte_days": 14,              # Minimum days to expiry
    "min_volume": 300,               # Minimum market volume
    "kelly_fraction": 0.25,          # Fractional Kelly
    "max_position": 0.10,            # Max position size per leg
    "max_event_exposure": 0.20,      # Max total exposure per event
    "min_pair_correlation": 0.60,    # Minimum historical price correlation
})

data_loader = DataLoader(DB_PATH)

print(f"\n{'='*60}")
print(f"Polymarket Backtest — Spread Dynamics Strategy")
print(f"{'='*60}")
print(f"Strategy:        {strategy.name}")
print(f"DB:              {DB_PATH}")
print(f"Initial Capital: $10,000")
print(f"{'='*60}\n")

engine = BacktestEngine(
    strategy=strategy,
    data_loader=data_loader,
    initial_capital=10_000.0,
    commission=0.0,
    max_position_size=0.1,   # Max 10% of capital per position
    max_positions=None,       # No limit on number of positions
    verbose=True,
)

results = engine.run(
    start_date="2024-01-01",
    end_date="2026-03-16",
    use_timing_markets=True,  # Use timing markets for term structure
    min_volume=300,
)

if not results:
    print("Backtest failed — no results.")
    sys.exit(1)

engine.print_results()

# Save trades
os.makedirs("backtest_results", exist_ok=True)
if not results["trades"].empty:
    out = "backtest_results/spread_dynamics_trades.csv"
    results["trades"].to_csv(out, index=False)
    print(f"\nTrades saved to: {out}")
else:
    print("\nNo trades executed.")
