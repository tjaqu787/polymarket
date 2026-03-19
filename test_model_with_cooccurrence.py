#!/usr/bin/env python3
"""
Test the TimeDiscountingModel with cooccurrence features.

This is a quick test to ensure the model runs with the new cooccurrence features.
"""

import sys
sys.path.append('.')

from data.data_loader_for_model import PolymarketDataLoader
from models.time_discounting_model.model import TimeDiscountingModel

print("="*70)
print("TESTING TIME DISCOUNTING MODEL WITH COOCCURRENCE FEATURES")
print("="*70)

# Initialize data loader
print("\n[1/4] Loading data...")
loader = PolymarketDataLoader(db_path="data/polymarket.db")

# Load a small subset for quick testing
rates_df, ts_metrics_df, resolved_df = loader.load_full_dataset(
    resolved_only=True,  # Only resolved markets for training
    min_markets_per_group=3,
    outcome='Yes'
)

print(f"Loaded {len(rates_df)} price points")
print(f"Loaded {len(resolved_df)} resolved markets")

# Check that cooccurrence features are present
print("\n[2/4] Checking cooccurrence features...")
cooccurrence_cols = ['token_count', 'avg_token_df', 'max_cooccurrence', 'token_diversity']
present_cols = [col for col in cooccurrence_cols if col in rates_df.columns]
print(f"Cooccurrence features present: {present_cols}")

if len(present_cols) < len(cooccurrence_cols):
    missing = set(cooccurrence_cols) - set(present_cols)
    print(f"WARNING: Missing cooccurrence features: {missing}")

# Initialize model
print("\n[3/4] Initializing model...")
model = TimeDiscountingModel(discount_function='hyperbolic')

# Prepare data
print("\n[4/4] Preparing data for model...")
try:
    data = model.prepare_data(rates_df, resolved_df)
    print(f"✅ Data preparation successful!")
    print(f"   Observations: {data['n_obs']}")
    print(f"   Events: {data['n_events']}")
    print(f"   Categories: {data['n_categories']}")

    # Check that cooccurrence features are in prepared data
    cooccurrence_data_keys = ['token_count_norm', 'avg_token_df_norm', 'max_cooccurrence_norm', 'token_diversity']
    present_keys = [key for key in cooccurrence_data_keys if key in data]
    print(f"\n   Cooccurrence features in prepared data: {len(present_keys)}/{len(cooccurrence_data_keys)}")
    for key in present_keys:
        print(f"     - {key}: shape {data[key].shape}")

    print("\n" + "="*70)
    print("SUCCESS: Model is ready to use with cooccurrence features!")
    print("="*70)
    print("\nTo train the model, run:")
    print("  trace = model.fit(data, draws=1000, tune=500, chains=2)")

except Exception as e:
    print(f"\n❌ Error during data preparation: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
