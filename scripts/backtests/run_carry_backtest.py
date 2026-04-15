#!/usr/bin/env python3
"""
Run backtest using the CarryStrategy.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from backtest import BacktestEngine, DataLoader
from backtest.strategies.carry_strategy import CarryStrategy

DB_PATH = "data/polymarket.db"

strategy = CarryStrategy(config={
    "max_days_to_expiry": 90,     # Only markets < 90 days to expiry
    "high_threshold": 0.95,        # Buy (long) when price > 0.90
    "low_threshold": 0.05,         # Short when price < 0.10
    "stop_loss_long": 0.9,        # Exit long if drops below 0.85
    "stop_loss_short": 0.1,       # Exit short if rises above 0.15
})

data_loader = DataLoader(DB_PATH)

print(f"\n{'='*60}")
print(f"Polymarket Backtest — Carry Strategy")
print(f"{'='*60}")
print(f"Strategy:       {strategy.name}")
print(f"DB:             {DB_PATH}")
print(f"Initial Capital: $10,000")
print(f"{'='*60}\n")

engine = BacktestEngine(
    strategy=strategy,
    data_loader=data_loader,
    initial_capital=10_000.0,
    commission=0.0,
    max_position_size=0.1,  # Max 10% of capital per position
    max_positions=None,      # No limit on number of positions
    verbose=True,
)

results = engine.run(
    start_date="2022-11-05",
    end_date="2026-03-16",
    use_timing_markets=False,  # Use all markets, not just timing markets
    min_volume=100,
)

if not results:
    print("Backtest failed — no results.")
    sys.exit(1)

engine.print_results()

# Save trades
os.makedirs("backtest_results", exist_ok=True)
if not results["trades"].empty:
    out = "backtest_results/carry_trades.csv"
    results["trades"].to_csv(out, index=False)
    print(f"Trades saved to: {out}")
else:
    print("No trades executed.")
