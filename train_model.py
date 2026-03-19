#!/usr/bin/env python3
"""
Train the Time Discounting Model with cooccurrence features.

This script:
1. Loads market data with cooccurrence features
2. Prepares data for the Bayesian model
3. Trains the model using MCMC sampling
4. Saves the trace and generates diagnostics
"""

import sys
import argparse
import pickle
from pathlib import Path
from datetime import datetime

import pandas as pd
import numpy as np

# Add project to path
sys.path.append('.')

from data.data_loader_for_model import PolymarketDataLoader
from models.time_discounting_model.model import TimeDiscountingModel

# Try to import PyMC (optional - will error later if needed)
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False


def main():
    parser = argparse.ArgumentParser(description='Train Time Discounting Model')
    parser.add_argument('--db', default='data/polymarket.db', help='Path to database')
    parser.add_argument('--output-dir', default='model_output', help='Output directory for results')
    parser.add_argument('--draws', type=int, default=2000, help='Number of MCMC samples')
    parser.add_argument('--tune', type=int, default=1000, help='Number of tuning steps')
    parser.add_argument('--chains', type=int, default=4, help='Number of MCMC chains')
    parser.add_argument('--target-accept', type=float, default=0.9, help='Target acceptance rate')
    parser.add_argument('--min-markets', type=int, default=3, help='Minimum markets per event group')
    parser.add_argument('--outcome', default='Yes', help='Outcome to model (Yes/No)')
    parser.add_argument('--discount', default='hyperbolic', choices=['hyperbolic', 'exponential'],
                       help='Discount function type')
    parser.add_argument('--dry-run', action='store_true', help='Only prepare data, do not train')

    args = parser.parse_args()

    # Check PyMC availability
    if not PYMC_AVAILABLE and not args.dry_run:
        print("ERROR: PyMC is not installed. Install with: pip install pymc arviz")
        print("Or run with --dry-run to only prepare data")
        sys.exit(1)

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    run_dir = output_dir / f"run_{timestamp}"
    run_dir.mkdir(exist_ok=True)

    print("="*70)
    print("TIME DISCOUNTING MODEL TRAINING")
    print("="*70)
    print(f"Output directory: {run_dir}")
    print(f"Database: {args.db}")
    print(f"Discount function: {args.discount}")
    print(f"MCMC settings: {args.draws} draws, {args.tune} tune, {args.chains} chains")
    print("="*70)

    # ========================================
    # 1. LOAD DATA
    # ========================================
    print("\n[1/5] Loading data...")
    loader = PolymarketDataLoader(db_path=args.db)

    rates_df, ts_metrics_df, resolved_df = loader.load_full_dataset(
        resolved_only=True,  # Only use resolved markets for training
        min_markets_per_group=args.min_markets,
        outcome=args.outcome
    )

    print(f"  Loaded {len(rates_df)} price points")
    print(f"  Loaded {len(resolved_df)} resolved markets")

    # Check cooccurrence features
    cooccurrence_cols = ['token_count', 'avg_token_df', 'max_cooccurrence', 'token_diversity']
    present_cols = [col for col in cooccurrence_cols if col in rates_df.columns]
    missing_cols = set(cooccurrence_cols) - set(present_cols)

    if missing_cols:
        print(f"  WARNING: Missing cooccurrence features: {missing_cols}")
        print(f"  Run: python data/build_slug_cooccurrence.py")
    else:
        print(f"  ✅ All cooccurrence features present")

    # Save raw data
    rates_df.to_parquet(run_dir / 'rates_data.parquet')
    resolved_df.to_parquet(run_dir / 'resolved_data.parquet')
    print(f"  Saved raw data to {run_dir}")

    # ========================================
    # 2. PREPARE DATA FOR MODEL
    # ========================================
    print("\n[2/5] Preparing data for model...")
    model = TimeDiscountingModel(discount_function=args.discount)

    try:
        data = model.prepare_data(rates_df, resolved_df)
        print(f"  ✅ Data preparation successful")
        print(f"     Observations: {data['n_obs']}")
        print(f"     Events: {data['n_events']}")
        print(f"     Categories: {data['n_categories']}")

        # Check cooccurrence features in prepared data
        coo_keys = ['token_count_norm', 'avg_token_df_norm', 'max_cooccurrence_norm', 'token_diversity']
        present_keys = [k for k in coo_keys if k in data]
        print(f"     Cooccurrence features: {len(present_keys)}/{len(coo_keys)}")

        # Save prepared data
        with open(run_dir / 'prepared_data.pkl', 'wb') as f:
            pickle.dump(data, f)
        print(f"  Saved prepared data to {run_dir / 'prepared_data.pkl'}")

    except Exception as e:
        print(f"  ❌ Error preparing data: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if args.dry_run:
        print("\n" + "="*70)
        print("DRY RUN COMPLETE - Data prepared successfully")
        print("="*70)
        print(f"To train the model, run without --dry-run flag")
        return

    # ========================================
    # 3. BUILD MODEL
    # ========================================
    print("\n[3/5] Building PyMC model...")
    try:
        pm_model = model.build_model(data)
        print(f"  ✅ Model built successfully")
        print(f"     Model variables: {len(pm_model.free_RVs)}")

    except Exception as e:
        print(f"  ❌ Error building model: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ========================================
    # 4. TRAIN MODEL
    # ========================================
    print("\n[4/5] Training model with MCMC...")
    print(f"  This may take a while...")
    print(f"  Sampling {args.draws} draws across {args.chains} chains...")

    try:
        trace = model.fit(
            data,
            draws=args.draws,
            tune=args.tune,
            chains=args.chains,
            target_accept=args.target_accept
        )
        print(f"  ✅ Sampling complete")

        # Save trace
        trace.to_netcdf(run_dir / 'trace.nc')
        print(f"  Saved trace to {run_dir / 'trace.nc'}")

    except Exception as e:
        print(f"  ❌ Error during sampling: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    # ========================================
    # 5. DIAGNOSTICS
    # ========================================
    print("\n[5/5] Generating diagnostics...")

    try:
        # Convergence diagnostics
        summary = az.summary(trace, var_names=['β_ts_level', 'β_ts_slope', 'β_implied_rate',
                                               'β_token_count', 'β_avg_token_df',
                                               'β_max_cooccurrence', 'β_token_diversity'])
        print("\nCoefficient Summary:")
        print(summary)

        # Save summary
        summary.to_csv(run_dir / 'summary.csv')

        # Check convergence
        rhat_max = summary['r_hat'].max()
        if rhat_max > 1.01:
            print(f"\n  ⚠️  WARNING: Poor convergence detected (max r_hat={rhat_max:.4f})")
            print(f"     Consider increasing --tune or --draws")
        else:
            print(f"\n  ✅ Good convergence (max r_hat={rhat_max:.4f})")

        # Effective sample size
        ess_min = summary['ess_bulk'].min()
        print(f"  Minimum effective sample size: {ess_min:.0f}")

        # Generate plots if possible
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            # Trace plots
            az.plot_trace(trace, var_names=['β_token_count', 'β_max_cooccurrence'])
            plt.savefig(run_dir / 'trace_plot.png', dpi=150, bbox_inches='tight')
            plt.close()

            print(f"  Saved diagnostic plots to {run_dir}")

        except ImportError:
            print("  (matplotlib not available for plots)")

    except Exception as e:
        print(f"  ⚠️  Error generating diagnostics: {e}")

    # ========================================
    # COMPLETE
    # ========================================
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"Results saved to: {run_dir}")
    print(f"\nOutput files:")
    print(f"  - trace.nc: MCMC samples")
    print(f"  - prepared_data.pkl: Model input data")
    print(f"  - rates_data.parquet: Raw price/rates data")
    print(f"  - summary.csv: Coefficient estimates")
    print(f"\nTo make predictions, run:")
    print(f"  python predict_model.py --trace {run_dir / 'trace.nc'}")


if __name__ == "__main__":
    main()
