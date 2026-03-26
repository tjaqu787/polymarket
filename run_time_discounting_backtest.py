#!/usr/bin/env python3
"""
Run backtest using the TimeDiscountingStrategy.

This strategy uses a hierarchical Bayesian model to predict event outcomes
based on category-specific priors, event-level parameters, and term structure signals.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from backtest import BacktestEngine, DataLoader
from backtest.strategies.time_discounting_strategy import TimeDiscountingStrategy

DB_PATH = "data/polymarket.db"

strategy = TimeDiscountingStrategy(config={
    "db_path": DB_PATH,
    "threshold": 0.15,         # Price difference threshold for signals
    "min_edge": 0.05,          # Minimum edge to trade
    "min_volume": 100,         # Minimum market volume
    "lookback_days": 365,      # Days of historical data for training
    "retrain_interval": 7,     # Retrain every 7 days
    "mcmc_draws": 1000,        # MCMC posterior draws (lower for speed)
    "mcmc_tune": 500,          # MCMC tuning steps
    "mcmc_chains": 2,          # Number of MCMC chains
    "max_event_exposure": 0.15,  # 15% of portfolio per semantic group
})

data_loader = DataLoader(DB_PATH)

print(f"\n{'='*60}")
print(f"Polymarket Backtest — Time Discounting Strategy")
print(f"{'='*60}")
print(f"Strategy:        {strategy.name}")
print(f"DB:              {DB_PATH}")
print(f"Period:          2025-11-05 to 2026-03-16  (full data range)")
print(f"Initial Capital: $10,000")
print(f"Model:           Hierarchical Bayesian (PyMC)")
print(f"  - Threshold:   {strategy.threshold}")
print(f"  - Min Edge:    {strategy.min_edge}")
print(f"  - Lookback:    {strategy.lookback_days} days")
print(f"  - Retrain:     every {strategy.retrain_interval} days")
print(f"  - MCMC:        {strategy.mcmc_draws} draws x {strategy.mcmc_chains} chains")
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
os.makedirs("backtest/backtest_output", exist_ok=True)
if not results["trades"].empty:
    out = "backtest/backtest_output/time_discounting_trades.csv"
    results["trades"].to_csv(out, index=False)
    print(f"Trades saved to: {out}")
else:
    print("No trades executed.")
