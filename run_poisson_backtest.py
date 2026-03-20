#!/usr/bin/env python3
"""
Run backtest using the PoissonTimingStrategy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from backtest import BacktestEngine, DataLoader
from backtest.strategies.poisson_timing_strategy import PoissonTimingStrategy

DB_PATH = "data/polymarket.db"

strategy = PoissonTimingStrategy(config={
    "ci_level": 0.70,  # 70% credible interval
    "min_buckets": 3,
    "max_rmse": 0.3,
    "distribution": "gamma",
    "refit_days": 7,
    "n_bootstrap": 500,  # Bootstrap samples for CI calculation
})

data_loader = DataLoader(DB_PATH)

print(f"\n{'='*60}")
print(f"Polymarket Backtest — Poisson Timing Strategy")
print(f"{'='*60}")
print(f"Strategy:       {strategy.name}")
print(f"DB:             {DB_PATH}")
print(f"Period:         2025-11-05 to 2026-03-16  (full data range)")
print(f"Initial Capital: $10,000")
print(f"{'='*60}\n")

engine = BacktestEngine(
    strategy=strategy,
    data_loader=data_loader,
    initial_capital=10_000.0,
    commission=0.0,
    max_position_size=0.1,
    max_positions=None,
    verbose=True,
)

results = engine.run(
    start_date="2025-11-05",
    end_date="2026-03-16",
    use_timing_markets=True,
    min_volume=100,
)

if not results:
    print("Backtest failed — no results.")
    sys.exit(1)

engine.print_results()

# Save trades
os.makedirs("backtest_results", exist_ok=True)
if not results["trades"].empty:
    out = "backtest_results/poisson_trades.csv"
    results["trades"].to_csv(out, index=False)
    print(f"Trades saved to: {out}")
else:
    print("No trades executed.")
