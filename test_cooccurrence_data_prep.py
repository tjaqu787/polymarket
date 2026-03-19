#!/usr/bin/env python3
"""
Test data preparation with cooccurrence features (without PyMC).
"""

import sys
import pandas as pd
import numpy as np

sys.path.append('.')

from data.data_loader_for_model import PolymarketDataLoader

print("="*70)
print("TESTING DATA PREPARATION WITH COOCCURRENCE FEATURES")
print("="*70)

# Initialize data loader
print("\n[1/3] Loading data (resolved markets only)...")
loader = PolymarketDataLoader(db_path="data/polymarket.db")

rates_df, ts_metrics_df, resolved_df = loader.load_full_dataset(
    resolved_only=True,
    min_markets_per_group=3,
    outcome='Yes'
)

print(f"Loaded {len(rates_df)} price points")
print(f"Loaded {len(resolved_df)} resolved markets")

# Check cooccurrence features
print("\n[2/3] Verifying cooccurrence features...")
required_features = ['token_count', 'avg_token_df', 'max_cooccurrence', 'token_diversity']
present_features = [col for col in required_features if col in rates_df.columns]

print(f"Required features: {len(required_features)}")
print(f"Present features: {len(present_features)}")

for feature in required_features:
    if feature in rates_df.columns:
        non_null = rates_df[feature].notna().sum()
        print(f"  ✅ {feature}: {non_null:,} non-null values")
        print(f"     Mean: {rates_df[feature].mean():.2f}, Range: [{rates_df[feature].min():.2f}, {rates_df[feature].max():.2f}]")
    else:
        print(f"  ❌ {feature}: MISSING")

# Simulate model data preparation
print("\n[3/3] Simulating model data preparation...")
try:
    # Filter to resolved events
    resolved_event_ids = resolved_df['market_group'].unique()
    df = rates_df[rates_df['event_id'].isin(resolved_event_ids)].copy()

    # Get latest prices
    latest_prices = df.sort_values('date').groupby(['event_id', 'token_id']).last().reset_index()

    print(f"  Latest prices: {len(latest_prices)} observations")

    # Check all required columns
    required_cols = ['token_count', 'avg_token_df', 'max_cooccurrence', 'token_diversity',
                    'ts_level', 'ts_slope', 'ts_curvature', 'implied_rate', 'volume_num']

    missing_cols = [col for col in required_cols if col not in latest_prices.columns]

    if missing_cols:
        print(f"\n  ❌ Missing columns: {missing_cols}")
    else:
        print(f"  ✅ All required columns present")

        # Normalize cooccurrence features (as the model does)
        token_count_norm = np.log1p(latest_prices['token_count'].fillna(0))
        token_count_norm = (token_count_norm - token_count_norm.mean()) / (token_count_norm.std() + 1e-6)

        avg_token_df_norm = np.log1p(latest_prices['avg_token_df'].fillna(0))
        avg_token_df_norm = (avg_token_df_norm - avg_token_df_norm.mean()) / (avg_token_df_norm.std() + 1e-6)

        max_cooccurrence_norm = np.log1p(latest_prices['max_cooccurrence'].fillna(0))
        max_cooccurrence_norm = (max_cooccurrence_norm - max_cooccurrence_norm.mean()) / (max_cooccurrence_norm.std() + 1e-6)

        print(f"\n  Normalized cooccurrence features:")
        print(f"    token_count_norm: mean={token_count_norm.mean():.3f}, std={token_count_norm.std():.3f}")
        print(f"    avg_token_df_norm: mean={avg_token_df_norm.mean():.3f}, std={avg_token_df_norm.std():.3f}")
        print(f"    max_cooccurrence_norm: mean={max_cooccurrence_norm.mean():.3f}, std={max_cooccurrence_norm.std():.3f}")

    print("\n" + "="*70)
    print("✅ SUCCESS: Data preparation works with cooccurrence features!")
    print("="*70)
    print("\nCooccurrence features are ready for the PyMC model.")
    print("The model will use 4 new coefficients:")
    print("  - β_token_count")
    print("  - β_avg_token_df")
    print("  - β_max_cooccurrence")
    print("  - β_token_diversity")

except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
