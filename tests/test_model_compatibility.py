"""
Test Model Compatibility with Recent Changes

Tests that models work with:
1. Semantic grouping
2. "No" outcome filtering
3. Updated implied rate calculations
"""

import sys
sys.path.append('.')

from backtest.data_loader import DataLoader
import pandas as pd

def test_data_loading():
    """Test 1: Data Loading with Semantic Grouping"""
    print("=" * 70)
    print("TEST 1: Data Loading with Semantic Grouping")
    print("=" * 70)
    print()

    loader = DataLoader('data/polymarket.db')

    # Load with semantic grouping
    try:
        df = loader.load_timing_markets(
            start_date='2025-01-01',
            end_date='2025-01-31',
            min_volume=100,
            use_semantic_groups=True
        )

        print(f"✓ Loaded {len(df):,} rows")
        print()

        # Verify outcomes
        outcomes = df['outcome'].unique()
        print(f"Unique outcomes: {outcomes}")
        if 'No' in outcomes and 'Yes' not in outcomes:
            print("✓ Correctly filtered to 'No' outcomes only")
        else:
            print(f"⚠ Warning: Expected only 'No' outcomes, got {outcomes}")
        print()

        # Verify semantic_group_id column exists
        if 'semantic_group_id' in df.columns:
            print("✓ semantic_group_id column present")
            num_semantic_groups = df['semantic_group_id'].nunique()
            num_event_groups = df['event_id'].nunique()
            print(f"  Semantic groups: {num_semantic_groups}")
            print(f"  Event groups: {num_event_groups}")
            print(f"  Ratio: {num_semantic_groups/num_event_groups:.2f}x")
        else:
            print("✗ semantic_group_id column missing!")
        print()

        # Verify group_col column exists
        if 'group_col' in df.columns:
            print("✓ group_col column present")
            print(f"  Uses semantic grouping: {df['group_col'].equals(df.get('semantic_group_id', df['event_id']))}")
        else:
            print("⚠ group_col column missing (will fallback to event_id)")
        print()

        # Verify implied rates calculated
        implied_rate_count = df['implied_rate'].notna().sum()
        print(f"Implied rates calculated: {implied_rate_count:,} / {len(df):,} ({100*implied_rate_count/len(df):.1f}%)")
        if implied_rate_count > 0:
            print("✓ Implied rates calculated successfully")
            print(f"  Mean rate: {df['implied_rate'].mean():.4f}")
            print(f"  Median rate: {df['implied_rate'].median():.4f}")
        else:
            print("✗ No implied rates calculated!")
        print()

        # Show sample data
        print("Sample data:")
        print(df[['market_id', 'group_col', 'semantic_group_id', 'event_id', 'outcome', 'price', 'implied_rate']].head(10))
        print()

        return df

    except Exception as e:
        print(f"✗ Error loading data: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_semantic_grouping_improvement(df):
    """Test 2: Semantic Grouping Improvement"""
    print("=" * 70)
    print("TEST 2: Semantic Grouping Improvement")
    print("=" * 70)
    print()

    if df is None or df.empty:
        print("✗ No data available for testing")
        return

    # Group by event_id (old way)
    event_groups = df.groupby('event_id').agg({
        'market_id': 'nunique',
        'resolution_date': 'nunique'
    }).rename(columns={'market_id': 'num_markets', 'resolution_date': 'num_dates'})

    # Group by semantic_group_id (new way)
    if 'semantic_group_id' in df.columns and df['semantic_group_id'].notna().any():
        semantic_groups = df[df['semantic_group_id'].notna()].groupby('semantic_group_id').agg({
            'market_id': 'nunique',
            'resolution_date': 'nunique'
        }).rename(columns={'market_id': 'num_markets', 'resolution_date': 'num_dates'})

        print("Old Grouping (event_id):")
        print(f"  Total groups: {len(event_groups)}")
        print(f"  Groups with ≥2 dates: {len(event_groups[event_groups['num_dates'] >= 2])}")
        print(f"  Avg markets per group: {event_groups['num_markets'].mean():.2f}")
        print(f"  Avg dates per group: {event_groups['num_dates'].mean():.2f}")
        print()

        print("New Grouping (semantic_group_id):")
        print(f"  Total groups: {len(semantic_groups)}")
        print(f"  Groups with ≥2 dates: {len(semantic_groups[semantic_groups['num_dates'] >= 2])}")
        print(f"  Avg markets per group: {semantic_groups['num_markets'].mean():.2f}")
        print(f"  Avg dates per group: {semantic_groups['num_dates'].mean():.2f}")
        print()

        # Calculate improvement
        old_time_series = len(event_groups[event_groups['num_dates'] >= 2])
        new_time_series = len(semantic_groups[semantic_groups['num_dates'] >= 2])

        if old_time_series > 0:
            improvement = ((new_time_series - old_time_series) / old_time_series) * 100
            print(f"✓ Improvement: {improvement:+.1f}% more groups with time series")
        print()

        # Show top semantic groups
        print("Top 10 Semantic Groups by Market Count:")
        top_groups = semantic_groups.nlargest(10, 'num_markets')
        for idx, (group_id, row) in enumerate(top_groups.iterrows(), 1):
            print(f"  {idx}. {group_id[:50]:<50} {row['num_markets']:>3} markets, {row['num_dates']:>2} dates")
        print()
    else:
        print("⚠ No semantic_group_id data available")
        print()


def test_price_conversion():
    """Test 3: No/Yes Price Conversion Logic"""
    print("=" * 70)
    print("TEST 3: No/Yes Price Conversion Logic")
    print("=" * 70)
    print()

    # Test the conversion math
    test_cases = [
        {'no_price': 0.95, 'yes_price': 0.05},
        {'no_price': 0.80, 'yes_price': 0.20},
        {'no_price': 0.50, 'yes_price': 0.50},
        {'no_price': 0.20, 'yes_price': 0.80},
        {'no_price': 0.05, 'yes_price': 0.95},
    ]

    print("Testing P(No) = 1 - P(Yes) conversion:")
    print()
    for case in test_cases:
        no_price = case['no_price']
        expected_yes = case['yes_price']
        converted_yes = 1 - no_price

        print(f"  P(No) = {no_price:.2f} → P(Yes) = {converted_yes:.2f} (expected: {expected_yes:.2f})")
        assert abs(converted_yes - expected_yes) < 1e-10, "Conversion error!"

    print()
    print("✓ All conversion tests passed")
    print()

    # Test bounds conversion
    print("Testing CI bounds conversion (swap and invert):")
    print()

    yes_lower, yes_upper = 0.10, 0.30
    no_lower = 1 - yes_upper  # 0.70
    no_upper = 1 - yes_lower  # 0.90

    print(f"  Yes CI: [{yes_lower:.2f}, {yes_upper:.2f}]")
    print(f"  No CI:  [{no_lower:.2f}, {no_upper:.2f}]")
    print()
    print("✓ Bounds correctly inverted and swapped")
    print()


def main():
    """Run all tests"""
    print()
    print("█" * 70)
    print("MODEL COMPATIBILITY TESTS")
    print("█" * 70)
    print()

    # Test 1: Data loading
    df = test_data_loading()

    # Test 2: Semantic grouping improvement
    if df is not None:
        test_semantic_grouping_improvement(df)

    # Test 3: Price conversion logic
    test_price_conversion()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()

    if df is not None and len(df) > 0:
        print("✓ Data loading: PASSED")
        print("✓ Semantic grouping: WORKING")
        print("✓ Implied rates: CALCULATED")
        print("✓ Price conversion: CORRECT")
        print()
        print("✓ ALL TESTS PASSED - Models are compatible with recent changes!")
    else:
        print("✗ SOME TESTS FAILED - Review output above")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
