#!/usr/bin/env python3
"""
Quick verification that the data loader fix works.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.data_loader_for_model import PolymarketDataLoader
from datetime import datetime, timedelta

DB_PATH = "data/polymarket.db"

print("\n" + "="*80)
print("DATA LOADER FIX VERIFICATION")
print("="*80)

loader = PolymarketDataLoader(DB_PATH)

# Test with a shorter date range
start_date = "2025-12-01"
end_date = "2026-01-01"

print(f"\nLoading data from {start_date} to {end_date}")
print("Using default outcome parameter (should now be 'No')")

# Call with default parameters - should now work
rates_df, ts_metrics_df, resolved_df = loader.load_full_dataset(
    resolved_only=False,
    start_date=start_date,
    end_date=end_date,
    use_semantic_groups=True,
    load_token_features=False
    # Note: NOT passing outcome, using default which is now 'No'
)

print(f"\n{'='*80}")
print(f"RESULTS:")
print(f"{'='*80}")

print(f"\n1. rates_df:")
print(f"   - Rows: {len(rates_df)}")
if len(rates_df) > 0:
    if 'outcome' in rates_df.columns:
        print(f"   - Outcomes: {rates_df['outcome'].unique()}")
    print(f"   - Sample columns: {list(rates_df.columns[:10])}")

print(f"\n2. ts_metrics_df:")
print(f"   - Rows: {len(ts_metrics_df)}")
if len(ts_metrics_df) > 0:
    print(f"   - Has term structure data: YES")
    print(f"   - Sample columns: {list(ts_metrics_df.columns[:5])}")
else:
    print(f"   - Has term structure data: NO (needs more markets per group)")

print(f"\n3. resolved_df:")
print(f"   - Rows: {len(resolved_df)}")
print(f"   - Has 'resolved_outcome': {'resolved_outcome' in resolved_df.columns}")
if 'resolved_outcome' in resolved_df.columns:
    non_null = resolved_df['resolved_outcome'].notna().sum()
    print(f"   - Non-null outcomes: {non_null}")

print(f"\n{'='*80}")
if len(resolved_df) > 0 and 'resolved_outcome' in resolved_df.columns:
    print("✓ FIX VERIFIED: Data loader returns correct structure")
    print("✓ resolved_df is in the 3rd position with 'resolved_outcome' column")
else:
    print("✗ Issue: resolved_df doesn't have resolved outcomes")

if len(ts_metrics_df) > 0:
    print("✓ BONUS: Term structure metrics are now calculated (using 'No' outcomes)")
else:
    print("  Note: Term structure metrics still 0 (may need more markets per group)")

print(f"{'='*80}")
