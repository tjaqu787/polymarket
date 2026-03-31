#!/usr/bin/env python3
"""
Run backtest using the FactoredGammaStrategy.

This strategy unifies:
- Gamma CDF fitting from PoissonTimingStrategy
- Empirical Bayes factor adjustments inspired by TimeDiscountingStrategy

Key improvements:
- Category-level priors (politics vs crypto vs sports)
- Term structure feature adjustments (slope, curvature, implied rate)
- Dynamic position sizing based on prediction confidence (interval width)

Configuration:
- Empirical Bayes holdout end: 2025-10-05 (1 month before backtest)
- Backtest period: 2025-11-05 to 2026-03-16
- Max event exposure: 15% of portfolio
- Credible interval: 70%
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from backtest import BacktestEngine, DataLoader
from backtest.strategies.factored_gamma_strategy import FactoredGammaStrategy

DB_PATH = "data/polymarket.db"

# Strategy configuration
config = {
    "db_path": DB_PATH,
    "ci_level": 0.70,              # 70% credible interval
    "min_buckets": 3,               # Min term structure points
    "max_rmse": 0.3,                # Max fit error threshold
    "refit_hours": 6,               # Refit model every 6 hours (adaptive to data)
    "n_bootstrap": 500,             # Bootstrap samples for CI calculation
    "max_event_exposure": 0.15,     # 15% of portfolio per event
    "eb_holdout_end_date": "2025-10-05",  # Empirical Bayes cutoff date
}

strategy = FactoredGammaStrategy(config=config)
data_loader = DataLoader(DB_PATH)

print(f"\n{'='*70}")
print(f"Polymarket Backtest — Factored Gamma Timing Strategy")
print(f"{'='*70}")
print(f"Strategy:              {strategy.name}")
print(f"DB:                    {DB_PATH}")
print(f"EB Holdout End:        {config['eb_holdout_end_date']}")
print(f"Initial Capital:       $10,000")
print(f"Max Event Exposure:    {config['max_event_exposure']*100:.0f}%")
print(f"CI Level:              {config['ci_level']*100:.0f}%")
print(f"Refit Interval:        {config['refit_hours']} hours")
print(f"Fitting Method:        MLE + Bootstrap (frequentist)")
print(f"{'='*70}\n")

engine = BacktestEngine(
    strategy=strategy,
    data_loader=data_loader,
    initial_capital=10_000.0,
    commission=0.0,
    max_position_size=0.1,  # Max 10% per position (will be further constrained by strategy)
    max_positions=None,
    verbose=True,
)

print("Starting backtest...\n")

results = engine.run(
    start_date="2022-11-05",
    end_date="2026-03-16",
    use_timing_markets=True,
    min_volume=100,
)

if not results:
    print("\n❌ Backtest failed — no results.")
    sys.exit(1)

print(f"\n{'='*70}")
print("BACKTEST RESULTS")
print(f"{'='*70}\n")

engine.print_results()

# Save trades
os.makedirs("backtest_results", exist_ok=True)
if not results["trades"].empty:
    out = "backtest_results/factored_gamma_trades.csv"
    results["trades"].to_csv(out, index=False)
    print(f"\n✓ Trades saved to: {out}")
else:
    print("\n⚠ No trades executed.")

# Save portfolio history
if "portfolio_history" in results and not results["portfolio_history"].empty:
    out_history = "backtest_results/factored_gamma_portfolio.csv"
    results["portfolio_history"].to_csv(out_history, index=False)
    print(f"✓ Portfolio history saved to: {out_history}")

print(f"\n{'='*70}")
print("COMPARISON TO BASELINE")
print(f"{'='*70}")
print("To compare with PoissonTimingStrategy, run:")
print("  python run_poisson_backtest.py")
print("\nKey metrics to compare:")
print("  - Sharpe Ratio (risk-adjusted returns)")
print("  - Win Rate (% profitable trades)")
print("  - Max Drawdown (downside risk)")
print("  - Profit Factor (gross profit / gross loss)")
print(f"{'='*70}\n")
