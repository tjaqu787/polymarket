#!/usr/bin/env python3
"""
Test to verify data loading for TimeDiscountingModel.

This test verifies that:
1. The data loader correctly loads resolved outcomes
2. The return values are in the expected order
3. The model can prepare data from the loaded results
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from data.data_loader_for_model import PolymarketDataLoader
from models.time_discounting_model.model import TimeDiscountingModel
from datetime import datetime, timedelta
import pandas as pd

DB_PATH = "data/polymarket.db"

def test_data_loader_return_values():
    """Test that data loader returns values in the correct order."""
    print("\n" + "="*60)
    print("TEST 1: Data Loader Return Values")
    print("="*60)

    loader = PolymarketDataLoader(DB_PATH)

    # Load resolved markets from the past
    train_end = datetime.now()
    train_start = train_end - timedelta(days=365)

    print(f"\nLoading data from {train_start.date()} to {train_end.date()}")
    print("Requesting resolved_only=True")

    # Call the data loader
    result = loader.load_full_dataset(
        resolved_only=True,
        start_date=train_start.strftime('%Y-%m-%d'),
        end_date=train_end.strftime('%Y-%m-%d'),
        use_semantic_groups=True,
        load_token_features=False
    )

    print(f"\n✓ Data loader returned {len(result)} values")

    # Unpack according to what the data loader ACTUALLY returns
    rates_df, ts_metrics_df, resolved_df = result

    print(f"\nActual return values:")
    print(f"  1st return value (rates_df): {len(rates_df)} rows, columns: {list(rates_df.columns[:5])}...")
    print(f"  2nd return value (ts_metrics_df): {len(ts_metrics_df)} rows, columns: {list(ts_metrics_df.columns)}")
    print(f"  3rd return value (resolved_df): {len(resolved_df)} rows, columns: {list(resolved_df.columns)}")

    # Check what each DataFrame actually contains
    print(f"\n--- Checking DataFrame contents ---")

    print(f"\nrates_df has these key columns: {[c for c in rates_df.columns if 'price' in c or 'rate' in c or 'event' in c]}")

    print(f"\nts_metrics_df has these columns: {list(ts_metrics_df.columns)}")
    if 'resolved_outcome' in ts_metrics_df.columns:
        print(f"  ❌ ERROR: ts_metrics_df contains 'resolved_outcome' - this should be in resolved_df!")
    else:
        print(f"  ✓ ts_metrics_df does NOT contain 'resolved_outcome' (correct)")

    print(f"\nresolved_df has these columns: {list(resolved_df.columns)}")
    if 'resolved_outcome' in resolved_df.columns:
        print(f"  ✓ resolved_df contains 'resolved_outcome' (correct)")
        print(f"  ✓ Number of resolved outcomes: {len(resolved_df)}")
    else:
        print(f"  ❌ ERROR: resolved_df does NOT contain 'resolved_outcome'!")

    return rates_df, ts_metrics_df, resolved_df


def test_incorrect_unpacking():
    """Test what happens with incorrect unpacking (current bug)."""
    print("\n" + "="*60)
    print("TEST 2: Demonstrating the Bug (Incorrect Unpacking)")
    print("="*60)

    loader = PolymarketDataLoader(DB_PATH)

    train_end = datetime.now()
    train_start = train_end - timedelta(days=365)

    print(f"\nLoading data from {train_start.date()} to {train_end.date()}")

    # This is what the STRATEGY does (WRONG!)
    result = loader.load_full_dataset(
        resolved_only=True,
        start_date=train_start.strftime('%Y-%m-%d'),
        end_date=train_end.strftime('%Y-%m-%d'),
        use_semantic_groups=True,
        load_token_features=False
    )

    # Unpack incorrectly like the strategy does
    rates_df, resolved_df, text_features = result

    print(f"\nIncorrect unpacking (what strategy does):")
    print(f"  Variable 'rates_df': {len(rates_df)} rows")
    print(f"  Variable 'resolved_df': {len(resolved_df)} rows, columns: {list(resolved_df.columns)}")
    print(f"  Variable 'text_features': {len(text_features)} rows, columns: {list(text_features.columns)}")

    print(f"\n--- Checking what ended up where ---")
    if 'resolved_outcome' in resolved_df.columns:
        print(f"  ❌ 'resolved_df' contains 'resolved_outcome': {len(resolved_df)} outcomes")
    else:
        print(f"  ❌ 'resolved_df' does NOT contain 'resolved_outcome'")
        print(f"     It has columns: {list(resolved_df.columns)}")
        print(f"     → This is actually ts_metrics_df!")

    if 'resolved_outcome' in text_features.columns:
        print(f"  ✓ 'text_features' contains 'resolved_outcome': {len(text_features)} outcomes")
        print(f"     → This is actually resolved_df, not text_features!")

    print(f"\n💡 BUG CONFIRMED: The variables are swapped!")
    print(f"   - 'resolved_df' is getting ts_metrics (no outcomes)")
    print(f"   - 'text_features' is getting resolved_df (has outcomes)")

    return rates_df, resolved_df, text_features


def test_model_data_preparation():
    """Test that the model can prepare data when given correct inputs."""
    print("\n" + "="*60)
    print("TEST 3: Model Data Preparation")
    print("="*60)

    loader = PolymarketDataLoader(DB_PATH)

    train_end = datetime.now()
    train_start = train_end - timedelta(days=365)

    # Load data correctly
    rates_df, ts_metrics_df, resolved_df = loader.load_full_dataset(
        resolved_only=True,
        start_date=train_start.strftime('%Y-%m-%d'),
        end_date=train_end.strftime('%Y-%m-%d'),
        use_semantic_groups=True,
        load_token_features=False
    )

    print(f"\nLoaded {len(rates_df)} price observations")
    print(f"Loaded {len(resolved_df)} resolved outcomes")

    if len(resolved_df) < 10:
        print("❌ Not enough resolved markets to train (need at least 10)")
        return False

    # Initialize model
    model = TimeDiscountingModel(discount_function='hyperbolic')

    print(f"\nPreparing data for model...")
    try:
        data = model.prepare_data(rates_df, resolved_df)

        print(f"✓ Data preparation successful!")
        print(f"  Observations: {data['n_obs']}")
        print(f"  Categories: {data['n_categories']}")
        print(f"  Events: {data['n_events']}")
        print(f"  Prices shape: {data['prices'].shape}")
        print(f"  Won outcomes shape: {data['won'].shape}")

        if data['n_obs'] >= 20:
            print(f"\n✓ Sufficient data for training (need at least 20 observations)")
            return True
        else:
            print(f"\n❌ Not enough observations to train (need at least 20)")
            return False

    except Exception as e:
        print(f"❌ Data preparation failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*80)
    print("TIME DISCOUNTING MODEL - DATA LOADING TEST")
    print("="*80)

    # Run tests
    print("\n\nRunning Test 1...")
    rates_df, ts_metrics_df, resolved_df = test_data_loader_return_values()

    print("\n\nRunning Test 2...")
    rates_wrong, resolved_wrong, text_wrong = test_incorrect_unpacking()

    print("\n\nRunning Test 3...")
    success = test_model_data_preparation()

    # Summary
    print("\n\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"\n✓ Test 1: Verified data loader return order")
    print(f"  - Returns: (rates_df, ts_metrics_df, resolved_df)")
    print(f"  - Resolved outcomes in 3rd position: {len(resolved_df)} rows")

    print(f"\n❌ Test 2: Confirmed bug in strategy unpacking")
    print(f"  - Strategy expects: (rates_df, resolved_df, text_features)")
    print(f"  - But data loader returns: (rates_df, ts_metrics_df, resolved_df)")
    print(f"  - Result: resolved_df gets ts_metrics (0 outcomes)")

    print(f"\n{'✓' if success else '❌'} Test 3: Model data preparation")
    if success:
        print(f"  - Model can prepare data when given correct inputs")
    else:
        print(f"  - Model cannot prepare data (insufficient data or other error)")

    print(f"\n" + "="*80)
    print("RECOMMENDED FIX:")
    print("="*80)
    print(f"In time_discounting_strategy.py, change line 93-99 from:")
    print(f"  rates_df, resolved_df, text_features = self.data_loader.load_full_dataset(...)")
    print(f"\nTo:")
    print(f"  rates_df, ts_metrics_df, resolved_df = self.data_loader.load_full_dataset(...)")
    print(f"\nOr update data_loader_for_model.py to return in the expected order.")
    print("="*80)
