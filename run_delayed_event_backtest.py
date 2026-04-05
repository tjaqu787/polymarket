"""
Run backtest for Delayed Event Arbitrage Strategy

Strategy: Buy short-dated No + long-dated Yes when:
- Short-dated No ≥ 90% (event unlikely soon)
- Long-dated Yes ≥ 80% (event likely eventually)
"""

import sys
from backtest.engine import BacktestEngine
from backtest.strategies.delayed_event_strategy import DelayedEventStrategy

# Database path
DB_PATH = "data/polymarket.db"

# Strategy configuration
config = {
    "db_path": DB_PATH,
    "short_days_threshold": 20,      # Max days to expiry for short leg
    "short_no_threshold": 0.90,      # Min No price for short leg (90%)
    "long_yes_threshold": 0.80,      # Min Yes price for long leg (80%)
    "max_event_exposure": 0.15,      # 15% of portfolio per event
}

print("="*70)
print("Polymarket Backtest — Delayed Event Arbitrage")
print("="*70)
print(f"Strategy:              DelayedEvent_No90_Yes80")
print(f"DB:                    {DB_PATH}")
print(f"Initial Capital:       $10,000")
print(f"Max Event Exposure:    {config['max_event_exposure']*100:.0f}%")
print(f"\nEntry Criteria:")
print(f"  Short-dated (≤{config['short_days_threshold']}d): No ≥ {config['short_no_threshold']*100:.0f}%")
print(f"  Long-dated:            Yes ≥ {config['long_yes_threshold']*100:.0f}%")
print(f"\nExit: When short-dated expires (close both legs)")
print("="*70)

# Create strategy
strategy = DelayedEventStrategy(**config)

# Create data loader
from backtest.data_loader import DataLoader
data_loader = DataLoader(DB_PATH)

# Create backtest engine
engine = BacktestEngine(
    strategy=strategy,
    data_loader=data_loader,
    initial_capital=10000,
    commission=0.0,
    max_positions=50
)

print("\nStarting backtest...\n")

results = engine.run(
    start_date="2025-10-01",  # Expanded test period (4 months)
    end_date="2026-01-31",
    use_carry_markets=True,  # Use all markets
    min_volume=100,
)

if not results:
    print("\n❌ Backtest failed — no results.")
    sys.exit(1)

print(f"\n{'='*70}")
print("BACKTEST RESULTS")
print(f"{'='*70}\n")

# Print results
engine.print_results()

print("\n✓ Trades saved to: backtest_results/delayed_event_trades.csv")

print(f"\n{'='*70}")
print("NOTES")
print(f"{'='*70}")
print("This strategy bets that events happen BETWEEN the near and far dates:")
print("  - Near No ≥ 90%: Event unlikely in next 20 days")
print("  - Far Yes ≥ 80%: Event likely eventually")
print("  - Profit: Collect on near No + far Yes pays when event happens")
print(f"{'='*70}\n")
