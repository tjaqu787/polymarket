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
# Import directly to avoid pymc dependency from other strategies
import importlib.util
spec = importlib.util.spec_from_file_location(
    "spread_dynamics_strategy",
    "backtest/strategies/spread_dynamics_strategy.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
SpreadDynamicsStrategy = module.SpreadDynamicsStrategy

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
    # Bayesian Kelly parameters
    "use_bayesian_kelly": True,      # Enable Bayesian position sizing
    "prior_edge_mean": 0.0,          # Prior belief: no edge initially
    "prior_edge_std": 0.05,          # Prior uncertainty (wide)
    "obs_std": 0.02,                 # Observation noise
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

# Bayesian Analysis
print("\n" + "="*60)
print("BAYESIAN POSTERIOR ANALYSIS")
print("="*60)

posterior_stats = strategy.get_posterior_stats()
if not posterior_stats.empty:
    print(f"\nNumber of pairs traded: {len(posterior_stats)}")
    print(f"\nPosterior Statistics:")
    print(f"  Avg posterior mean:     {posterior_stats['posterior_mean'].mean():.6f}")
    print(f"  Avg posterior std:      {posterior_stats['posterior_std'].mean():.6f}")
    print(f"  Avg prior std:          {posterior_stats['prior_std'].mean():.6f}")
    print(f"  Avg uncertainty reduction: {posterior_stats['uncertainty_reduction'].mean()*100:.1f}%")
    print(f"  Total observations:     {posterior_stats['n_observations'].sum()}")

    print(f"\nTop 10 pairs by certainty (lowest posterior std):")
    top_certain = posterior_stats.nsmallest(10, 'posterior_std')[
        ['pair_id', 'posterior_mean', 'posterior_std', 'n_observations']
    ]
    print(top_certain.to_string(index=False))

    print(f"\nTop 10 pairs by positive edge (highest posterior mean):")
    top_edge = posterior_stats.nlargest(10, 'posterior_mean')[
        ['pair_id', 'posterior_mean', 'posterior_std', 'n_observations']
    ]
    print(top_edge.to_string(index=False))

    # Save posterior analysis
    posterior_stats.to_csv("backtest_results/bayesian_posteriors.csv", index=False)
    print(f"\nBayesian posteriors saved to: backtest_results/bayesian_posteriors.csv")
else:
    print("\nNo posterior data available.")

# Signal-to-Noise Analysis
print("\n" + "="*60)
print("SIGNAL-TO-NOISE METRICS")
print("="*60)

snr_metrics = strategy.get_signal_to_noise_metrics()
if snr_metrics:
    print(f"\nTrading Frequency:")
    print(f"  Total signals evaluated:    {snr_metrics['total_signals']}")
    print(f"  Total trades executed:      {snr_metrics['total_trades']}")
    print(f"  Signal-to-trade ratio:      {snr_metrics['signal_to_trade_ratio']:.2%}")
    print(f"  Trades per day:             {snr_metrics['trades_per_day']:.2f}")
    print(f"  Active trading days:        {snr_metrics['active_trading_days']}")
    print(f"  Avg trades/active day:      {snr_metrics['avg_trades_per_active_day']:.2f}")
    print(f"  Max trades in one day:      {snr_metrics['max_trades_per_day']}")
    print(f"  Unique pairs traded:        {snr_metrics['unique_pairs_traded']}")

    print(f"\nSignal Quality (higher = better):")
    print(f"  Velocity SNR:               {snr_metrics['velocity_snr']:.3f}")
    print(f"  Volume Z-score SNR:         {snr_metrics['vol_zscore_snr']:.3f}")
    print(f"  Avg edge confidence:        {snr_metrics['avg_edge_confidence']:.3f}")
    print(f"  Max edge confidence:        {snr_metrics['max_edge_confidence']:.3f}")

    print(f"\nRegime-Specific Stats:")
    for regime, stats in snr_metrics['regime_stats'].items():
        print(f"  {regime.upper()}:")
        print(f"    Signals:        {stats['total_signals']}")
        print(f"    Trades:         {stats['total_trades']}")
        print(f"    Trade ratio:    {stats['signal_to_trade_ratio']:.2%}")
        print(f"    Avg velocity:   {stats['avg_velocity']:.4f}")
        print(f"    Avg vol z:      {stats['avg_vol_zscore']:.2f}")

    print(f"\n  INTERPRETATION:")
    print(f"  - If signal-to-trade ratio is high (>80%), you're trading most signals → may be overtrading")
    print(f"  - If trades/day is high (>5), consider tightening filters")
    print(f"  - Low SNR (<1.0) means noise dominates signal → need stronger filters")
    print(f"  - Edge confidence <1.0 means high uncertainty → reduce position sizes")
else:
    print("\nNo signal data available yet.")

print("\n" + "="*60 + "\n")

# Save trades
os.makedirs("backtest_results", exist_ok=True)
if not results["trades"].empty:
    out = "backtest_results/spread_dynamics_trades.csv"
    results["trades"].to_csv(out, index=False)
    print(f"Trades saved to: {out}")
else:
    print("No trades executed.")
