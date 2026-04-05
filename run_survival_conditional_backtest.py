#!/usr/bin/env python3
"""
Run backtest using the Survival Conditional Strategy.

Strategy: Trade conditional (survival-adjusted) probability mispricing.

Markets quote P(event by T). As time passes without resolution, the correct
price should be the conditional probability:
    P(event in [T1, T2] | not yet) = [F(T2) - F(T1)] / [1 - F(T1)]

Markets are slow to update this. We exploit the gap.

Key features:
- Fit CDF to term structure (reusing existing Gamma fitter)
- Compute conditional fair value as event non-occurrence accumulates
- Trade when edge > min_survival_edge
- Roll forward every roll_forward_days to update conditional probabilities
- Exit when edge collapses or contract near expiry

Configuration:
- Min survival edge: 4% (minimum gap to trade)
- Kelly fraction: 25% (fractional Kelly sizing)
- Max position: 10% (per contract)
- Refit interval: 7 days
- Roll forward interval: 7 days
- Distribution: Gamma CDF
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from backtest import BacktestEngine, DataLoader
from backtest.strategies.survival_conditional_strategy import SurvivalConditionalStrategy

DB_PATH = "data/polymarket.db"

# Strategy configuration
config = {
    "db_path": DB_PATH,
    "min_tte_days": 7,              # Min days to expiry
    "max_tte_days": 365,            # Max days to expiry
    "refit_days": 7,                # Refit model every 7 days
    "min_survival_edge": 0.02,      # Min 2% edge to trade (lowered from 4%)
    "min_volume": 100,              # Min market volume (lowered from 500)
    "kelly_fraction": 0.25,         # 25% fractional Kelly
    "max_position": 0.10,           # Max 10% per position
    "max_group_exposure": 0.15,     # Max 15% per event group
    "distribution": "gamma",        # Use Gamma CDF
    "roll_forward_days": 7,         # Re-evaluate positions every 7 days
}

strategy = SurvivalConditionalStrategy(config=config)
data_loader = DataLoader(DB_PATH)

print(f"\n{'='*70}")
print(f"Polymarket Backtest — Survival Conditional Strategy")
print(f"{'='*70}")
print(f"Strategy:              {strategy.name()}")
print(f"DB:                    {DB_PATH}")
print(f"Initial Capital:       $10,000")
print(f"Min Survival Edge:     {config['min_survival_edge']*100:.0f}%")
print(f"Kelly Fraction:        {config['kelly_fraction']*100:.0f}%")
print(f"Refit Interval:        {config['refit_days']} days")
print(f"Roll Forward:          {config['roll_forward_days']} days")
print(f"Distribution:          {config['distribution']}")
print(f"{'='*70}\n")

engine = BacktestEngine(
    strategy=strategy,
    data_loader=data_loader,
    initial_capital=10_000.0,
    commission=0.0,
    max_position_size=0.1,  # Max 10% per position
    max_positions=None,
    verbose=True,
)

print("Starting backtest...\n")

results = engine.run(
    start_date="2024-11-01",  # 2-month test period
    end_date="2024-12-31",
    use_timing_markets=False,
    use_carry_markets=True,  # Use carry markets (all markets with semantic groups)
    min_volume=100,
)

if not results:
    print("\n❌ Backtest failed — no results.")
    sys.exit(1)

print(f"\n{'='*70}")
print("BACKTEST RESULTS")
print(f"{'='*70}\n")

# Print strategy statistics
strategy.on_backtest_end()

# Print performance results
engine.print_results()

# Save trades
os.makedirs("backtest_results", exist_ok=True)
if not results["trades"].empty:
    out = "backtest_results/survival_conditional_trades.csv"
    results["trades"].to_csv(out, index=False)
    print(f"\n✓ Trades saved to: {out}")
else:
    print("\n⚠ No trades executed.")

# Save portfolio history
if "portfolio_history" in results and not results["portfolio_history"].empty:
    out_history = "backtest_results/survival_conditional_portfolio.csv"
    results["portfolio_history"].to_csv(out_history, index=False)
    print(f"✓ Portfolio history saved to: {out_history}")

print(f"\n{'='*70}")
print("STRATEGY EXPLANATION")
print(f"{'='*70}")
print("This strategy exploits dynamic mispricing that compounds over time.")
print("As each week passes without an event, markets should compress the")
print("remaining probability mass onto shorter windows — but often don't.")
print("")
print("Edge = Market Price - Conditional Fair Value")
print("where Conditional Fair Value = [F(T2) - F(T1)] / [1 - F(T1)]")
print("")
print("Unlike static carry strategies, this exploits how event non-occurrence")
print("accumulates as new information that markets are slow to incorporate.")
print(f"{'='*70}\n")
