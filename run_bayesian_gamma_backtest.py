#!/usr/bin/env python3
"""
Run backtest using the BayesianGammaStrategy.

TRUE BAYESIAN SEQUENTIAL UPDATING:
- Uses PyMC MCMC instead of MLE + bootstrap
- Previous posterior → new prior (sequential belief updating)
- Multithreaded MCMC sampling for speed (uses all available cores)

Key Differences from Factored Gamma:
- Factored: MLE + bootstrap (frequentist, refits from scratch)
- Bayesian: MCMC + sequential (uses previous fit as informative prior)

Configuration:
- Refit every 6 hours (sequential Bayesian updating)
- MCMC: 500 draws × 2 chains = 1000 posterior samples
- Multithreaded: 12 cores for parallel MCMC sampling
- Backtest period: 2025-11-05 to 2026-03-16 (same as Factored Gamma)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from backtest import BacktestEngine, DataLoader
from backtest.strategies.bayesian_gamma_strategy import BayesianGammaStrategy

DB_PATH = "data/polymarket.db"

# Strategy configuration
config = {
    "db_path": DB_PATH,
    "ci_level": 0.70,              # 70% credible interval
    "min_buckets": 3,               # Min term structure points
    "mcmc_draws": 500,              # Posterior samples per chain
    "mcmc_tune": 500,               # MCMC tuning steps
    "mcmc_chains": 2,               # Number of MCMC chains
    "mcmc_cores": 12,               # USE ALL 12 CORES for parallel MCMC!
    "refit_hours": 6,               # Refit every 6 hours (sequential updating)
    "max_event_exposure": 0.15,     # 15% of portfolio per event
    "posterior_dir": "models/bayesian_gamma_model/posteriors",
}

strategy = BayesianGammaStrategy(config=config)
data_loader = DataLoader(DB_PATH)

print(f"\n{'='*70}")
print(f"Polymarket Backtest — Bayesian Gamma Timing Strategy")
print(f"{'='*70}")
print(f"Strategy:              {strategy.name}")
print(f"DB:                    {DB_PATH}")
print(f"Backtest Period:       2025-11-05 to 2026-03-16")
print(f"Initial Capital:       $10,000")
print(f"Max Event Exposure:    {config['max_event_exposure']*100:.0f}%")
print(f"CI Level:              {config['ci_level']*100:.0f}%")
print(f"Refit Interval:        {config['refit_hours']} hours")
print(f"Fitting Method:        PyMC MCMC (sequential Bayesian)")
print(f"MCMC Config:           {config['mcmc_draws']} draws × {config['mcmc_chains']} chains")
print(f"CPU Cores:             {config['mcmc_cores']} (multithreaded)")
print(f"Sequential Updating:   ✓ Previous posterior → new prior")
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

print("Starting backtest with Bayesian sequential updating...\n")
print("Note: MCMC sampling is slower than MLE but provides proper uncertainty quantification")
print("      First fits use weak priors, subsequent fits use previous posteriors\n")

results = engine.run(
    start_date="2025-11-05",
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
    out = "backtest_results/bayesian_gamma_trades.csv"
    results["trades"].to_csv(out, index=False)
    print(f"\n✓ Trades saved to: {out}")
else:
    print("\n⚠ No trades executed.")

print(f"\n{'='*70}")
print("COMPARISON TO FACTORED GAMMA (FREQUENTIST)")
print(f"{'='*70}")
print("Compare with:")
print("  python run_factored_gamma_backtest.py")
print("\nKey Questions:")
print("  - Does sequential Bayesian updating improve performance?")
print("  - Are credible intervals better calibrated than bootstrap CI?")
print("  - Does posterior narrowing (learning) help as events approach?")
print("  - Is the MCMC overhead worth the improved uncertainty quantification?")
print(f"{'='*70}\n")

print("Bayesian vs Frequentist:")
print("  Bayesian: MCMC posterior samples, sequential updating (prior → posterior)")
print("  Frequentist: MLE point estimates, bootstrap CI, refits from scratch")
print("\nExpected Benefits:")
print("  - More realistic uncertainty quantification")
print("  - Narrowing CI as event approaches (learning)")
print("  - Principled incorporation of previous information")
print("\nExpected Costs:")
print("  - Slower fitting (MCMC vs MLE)")
print("  - More complex implementation")
print("  - Storage overhead (posterior samples)")
print(f"\n{'='*70}\n")
