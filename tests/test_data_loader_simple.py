#!/usr/bin/env python3
"""
Simple test to verify data loading return values.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.data_loader_for_model import PolymarketDataLoader
from datetime import datetime, timedelta

DB_PATH = "data/polymarket.db"

print("\n" + "="*80)
print("DATA LOADER RETURN VALUE TEST")
print("="*80)

loader = PolymarketDataLoader(DB_PATH)

# Test with a recent date range
train_end = datetime.now()
train_start = train_end - timedelta(days=365)

print(f"\nLoading data from {train_start.date()} to {train_end.date()}")
print("Requesting: resolved_only=True")

# Call the data loader
result = loader.load_full_dataset(
    resolved_only=True,
    start_date=train_start.strftime('%Y-%m-%d'),
    end_date=train_end.strftime('%Y-%m-%d'),
    use_semantic_groups=True,
    load_token_features=False
)

print(f"\n{'='*80}")
print(f"RESULT: Data loader returned {len(result)} values")
print(f"{'='*80}")

# Unpack according to what the code SAYS it returns
rates_df, ts_metrics_df, resolved_df = result

print(f"\n1st return value (rates_df):")
print(f"  - Rows: {len(rates_df)}")
print(f"  - Sample columns: {list(rates_df.columns[:10])}")
print(f"  - Has 'resolved_outcome': {'resolved_outcome' in rates_df.columns}")

print(f"\n2nd return value (ts_metrics_df):")
print(f"  - Rows: {len(ts_metrics_df)}")
print(f"  - Columns: {list(ts_metrics_df.columns)}")
print(f"  - Has 'resolved_outcome': {'resolved_outcome' in ts_metrics_df.columns}")

print(f"\n3rd return value (resolved_df):")
print(f"  - Rows: {len(resolved_df)}")
print(f"  - Columns: {list(resolved_df.columns)}")
print(f"  - Has 'resolved_outcome': {'resolved_outcome' in resolved_df.columns}")

print(f"\n{'='*80}")
print("BUG DEMONSTRATION - What the strategy does:")
print("="*80)

# Now unpack it the WRONG way (like the strategy does)
rates_df2, resolved_df2, text_features2 = result

print(f"\nWhen unpacking as: rates_df, resolved_df, text_features = load_full_dataset(...)")
print(f"\nVariable 'resolved_df' receives:")
print(f"  - Rows: {len(resolved_df2)}")
print(f"  - Columns: {list(resolved_df2.columns)}")
print(f"  - Has 'resolved_outcome': {'resolved_outcome' in resolved_df2.columns}")

if 'resolved_outcome' not in resolved_df2.columns:
    print(f"\n❌ BUG CONFIRMED!")
    print(f"   The variable 'resolved_df' does NOT contain resolved outcomes!")
    print(f"   It's actually getting ts_metrics_df which has {len(resolved_df2)} rows")
    print(f"   The actual resolved outcomes are in the 3rd position (text_features)")

print(f"\nVariable 'text_features' receives:")
print(f"  - Rows: {len(text_features2)}")
print(f"  - Columns: {list(text_features2.columns)}")
print(f"  - Has 'resolved_outcome': {'resolved_outcome' in text_features2.columns}")

if 'resolved_outcome' in text_features2.columns:
    print(f"\n   The actual resolved_df ended up in 'text_features'!")
    print(f"   Number of resolved outcomes: {len(text_features2)}")

print(f"\n{'='*80}")
print("SUMMARY:")
print("="*80)
print(f"✓ Data loader returns: (rates_df, ts_metrics_df, resolved_df)")
print(f"✓ Resolved outcomes are in position 3: {len(resolved_df)} rows")
print(f"\n❌ Strategy expects: (rates_df, resolved_df, text_features)")
print(f"❌ This causes resolved_df to get ts_metrics_df: {len(resolved_df2)} rows with no outcomes")
print(f"\n{'='*80}")
print("FIX:")
print("="*80)
print("In time_discounting_strategy.py line 93, change:")
print("  rates_df, resolved_df, text_features = ...")
print("To:")
print("  rates_df, ts_metrics_df, resolved_df = ...")
print("="*80)
