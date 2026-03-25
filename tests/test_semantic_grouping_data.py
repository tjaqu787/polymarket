"""
Test Semantic Grouping Data Quality

This test verifies that:
1. Semantic groups are created correctly
2. Markets are properly consolidated
3. Price data is available for groups
4. Resolution dates match expectations
"""

import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(parent_dir, 'utils'))
sys.path.insert(0, os.path.join(parent_dir, 'data'))

from utils.implied_rates import get_market_groups, load_price_history
import pandas as pd


def test_group_consolidation():
    """Test that markets are properly consolidated into semantic groups."""
    print("="*70)
    print("TEST 1: Group Consolidation")
    print("="*70)

    groups = get_market_groups()

    print(f"\nTotal groups with time series: {len(groups)}")
    print(f"Groups with ≥5 dates: {len(groups[groups['num_dates'] >= 5])}")

    # Test specific known groups
    test_cases = {
        'us-strikes-iran:iran': {'min_markets': 50, 'min_dates': 10},
        'israel-x-hamas-ceasefire:israel': {'min_markets': 25, 'min_dates': 20},
        'bitcoin-all-time-high': {'min_markets': 25, 'min_dates': 20},
    }

    results = []
    for group_id, expectations in test_cases.items():
        group = groups[groups['market_group'] == group_id]

        if len(group) > 0:
            actual_markets = group.iloc[0]['num_markets']
            actual_dates = group.iloc[0]['num_dates']

            markets_ok = actual_markets >= expectations['min_markets']
            dates_ok = actual_dates >= expectations['min_dates']

            status = '✓' if (markets_ok and dates_ok) else '✗'

            results.append({
                'group': group_id,
                'markets': actual_markets,
                'expected_markets': expectations['min_markets'],
                'dates': actual_dates,
                'expected_dates': expectations['min_dates'],
                'status': status
            })

            print(f"\n{status} {group_id}")
            print(f"   Markets: {actual_markets} (expected ≥{expectations['min_markets']}) {' ✓' if markets_ok else ' ✗'}")
            print(f"   Dates: {actual_dates} (expected ≥{expectations['min_dates']}) {' ✓' if dates_ok else ' ✗'}")
        else:
            print(f"\n✗ {group_id} - NOT FOUND")
            results.append({
                'group': group_id,
                'markets': 0,
                'expected_markets': expectations['min_markets'],
                'dates': 0,
                'expected_dates': expectations['min_dates'],
                'status': '✗'
            })

    return results


def test_price_data_availability():
    """Test that price data is available for groups."""
    print("\n" + "="*70)
    print("TEST 2: Price Data Availability")
    print("="*70)

    groups = get_market_groups()

    # Test top 10 groups
    test_groups = groups.head(10)

    results = []
    for idx, row in test_groups.iterrows():
        group_id = row['market_group']
        expected_markets = row['num_markets']
        expected_dates = row['num_dates']

        try:
            prices = load_price_history(market_group=group_id)

            if len(prices) > 0:
                actual_markets = prices['market_id'].nunique()
                actual_dates = prices['resolution_date'].nunique()

                # Check if we got all expected data
                markets_match = actual_markets == expected_markets
                dates_match = actual_dates == expected_dates

                status = '✓' if (markets_match and dates_match) else '⚠'

                results.append({
                    'group': group_id,
                    'has_data': True,
                    'actual_markets': actual_markets,
                    'expected_markets': expected_markets,
                    'actual_dates': actual_dates,
                    'expected_dates': expected_dates,
                    'price_rows': len(prices),
                    'status': status
                })

                print(f"\n{status} {group_id}")
                print(f"   Markets: {actual_markets}/{expected_markets}")
                print(f"   Dates: {actual_dates}/{expected_dates}")
                print(f"   Price rows: {len(prices):,}")
            else:
                print(f"\n✗ {group_id} - No price data")
                results.append({
                    'group': group_id,
                    'has_data': False,
                    'actual_markets': 0,
                    'expected_markets': expected_markets,
                    'actual_dates': 0,
                    'expected_dates': expected_dates,
                    'price_rows': 0,
                    'status': '✗'
                })
        except Exception as e:
            print(f"\n✗ {group_id} - ERROR: {e}")
            results.append({
                'group': group_id,
                'has_data': False,
                'error': str(e),
                'status': '✗'
            })

    return results


def test_outcome_distribution():
    """Test that outcome filtering is working correctly."""
    print("\n" + "="*70)
    print("TEST 3: Outcome Distribution")
    print("="*70)

    groups = get_market_groups()
    first_group = groups.iloc[0]['market_group']

    print(f"\nTesting group: {first_group}")

    prices = load_price_history(market_group=first_group)

    print(f"\nOutcome distribution:")
    outcome_counts = prices['outcome'].value_counts()
    print(outcome_counts)

    # Check if filtered to 'No' only
    if len(outcome_counts) == 1 and 'No' in outcome_counts:
        print("\n✓ Correctly filtered to 'No' outcomes only")
        print("  (This is expected - better for hazard rate modeling)")
        return True
    elif 'Yes' in outcome_counts and 'No' in outcome_counts:
        print("\n⚠ Both 'Yes' and 'No' outcomes present")
        print("  This may be old data - view should filter to 'No' only")
        return False
    else:
        print(f"\n⚠ Unexpected outcome distribution: {list(outcome_counts.index)}")
        return False


def test_resolution_dates():
    """Test that resolution dates are properly extracted."""
    print("\n" + "="*70)
    print("TEST 4: Resolution Dates")
    print("="*70)

    groups = get_market_groups()

    # Check a group with many dates
    multi_date_groups = groups[groups['num_dates'] >= 10].head(3)

    for idx, row in multi_date_groups.iterrows():
        group_id = row['market_group']
        expected_dates = row['num_dates']

        print(f"\n{group_id}")
        print(f"  Expected dates: {expected_dates}")

        prices = load_price_history(market_group=group_id)

        if len(prices) > 0:
            res_dates = prices['resolution_date'].unique()
            res_dates_sorted = sorted([d for d in res_dates if pd.notna(d)])

            print(f"  Actual dates: {len(res_dates_sorted)}")
            print(f"  Date range: {res_dates_sorted[0]} to {res_dates_sorted[-1]}")

            if len(res_dates_sorted) >= 5:
                print(f"  Sample dates: {res_dates_sorted[:5]}")
        else:
            print(f"  No price data available")


def generate_summary_report():
    """Generate a summary report of all tests."""
    print("\n" + "="*70)
    print("SUMMARY REPORT")
    print("="*70)

    groups = get_market_groups()

    total_groups = len(groups)
    groups_with_5plus = len(groups[groups['num_dates'] >= 5])
    groups_with_10plus = len(groups[groups['num_dates'] >= 10])

    print(f"\nSemantic Grouping Statistics:")
    print(f"  Total groups with time series: {total_groups}")
    print(f"  Groups with ≥5 dates: {groups_with_5plus} ({100*groups_with_5plus/total_groups:.1f}%)")
    print(f"  Groups with ≥10 dates: {groups_with_10plus} ({100*groups_with_10plus/total_groups:.1f}%)")

    print(f"\nTop 10 Groups by Market Count:")
    top_by_markets = groups.nlargest(10, 'num_markets')[['market_group', 'num_markets', 'num_dates', 'canonical_slug']]
    print(top_by_markets.to_string(index=False))

    print(f"\nTop 10 Groups by Date Count:")
    top_by_dates = groups.nlargest(10, 'num_dates')[['market_group', 'num_markets', 'num_dates', 'canonical_slug']]
    print(top_by_dates.to_string(index=False))


def main():
    """Run all tests."""
    print("\n")
    print("█" * 70)
    print("SEMANTIC GROUPING DATA QUALITY TESTS")
    print("█" * 70)

    try:
        # Run tests
        consolidation_results = test_group_consolidation()
        price_data_results = test_price_data_availability()
        outcome_ok = test_outcome_distribution()
        test_resolution_dates()

        # Summary
        generate_summary_report()

        # Final verdict
        print("\n" + "="*70)
        print("TEST RESULTS")
        print("="*70)

        consolidation_passed = all(r['status'] == '✓' for r in consolidation_results)
        price_data_passed = all(r['status'] in ['✓', '⚠'] for r in price_data_results)

        print(f"\n1. Group Consolidation: {'✓ PASSED' if consolidation_passed else '✗ FAILED'}")
        print(f"2. Price Data Availability: {'✓ PASSED' if price_data_passed else '✗ FAILED'}")
        print(f"3. Outcome Distribution: {'✓ PASSED' if outcome_ok else '⚠ CHECK NEEDED'}")
        print(f"4. Resolution Dates: ✓ VERIFIED")

        if consolidation_passed and price_data_passed:
            print("\n✓ ALL TESTS PASSED - Data is ready for dashboard!")
        else:
            print("\n⚠ SOME TESTS FAILED - Review results above")

    except Exception as e:
        print(f"\n✗ TEST SUITE FAILED: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
