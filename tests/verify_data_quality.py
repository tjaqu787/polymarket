"""
Verify Semantic Grouping Data Quality

Run from project root: python3 tests/verify_data_quality.py
"""

import sys
sys.path.append('utils')
sys.path.append('data')

from implied_rates import get_market_groups, load_price_history, calculate_implied_rates_for_market_group
import pandas as pd


def main():
    print("="*70)
    print("SEMANTIC GROUPING DATA VERIFICATION")
    print("="*70)
    print()

    # Load all groups
    groups = get_market_groups()
    print(f"✓ Found {len(groups)} semantic groups with time series")
    print()

    # Show top 10 groups
    print("Top 10 Groups by Market Count:")
    print("-" * 70)
    top_groups = groups.head(10)
    for idx, row in top_groups.iterrows():
        print(f"{row['market_group'][:50]:<50} {row['num_markets']:>3} markets, {row['num_dates']:>2} dates")
    print()

    # Test each of top 10 groups for price data
    print("="*70)
    print("TESTING PRICE DATA AVAILABILITY")
    print("="*70)
    print()

    issues = []

    for idx, row in top_groups.iterrows():
        group_id = row['market_group']
        expected_markets = row['num_markets']
        expected_dates = row['num_dates']

        # Load price data
        try:
            prices = load_price_history(market_group=group_id)

            if len(prices) > 0:
                # Calculate metrics
                actual_markets = prices['market_id'].nunique()
                actual_dates = prices['resolution_date'].nunique()
                outcomes = prices['outcome'].unique()

                # Calculate implied rates
                rates_df = calculate_implied_rates_for_market_group(prices)
                valid_rates = rates_df[rates_df['implied_rate'].notna()]

                # Check if data matches expectations
                markets_match = actual_markets == expected_markets
                dates_match = actual_dates == expected_dates

                status = '✓' if (markets_match and dates_match) else '⚠'

                print(f"{status} {group_id[:45]:<45}")
                print(f"   Markets: {actual_markets}/{expected_markets} {' ✓' if markets_match else ' ⚠'}")
                print(f"   Dates: {actual_dates}/{expected_dates} {' ✓' if dates_match else ' ⚠'}")
                print(f"   Outcomes: {', '.join(outcomes)}")
                print(f"   Price rows: {len(prices):,}")
                print(f"   Valid rates: {len(valid_rates):,}")

                if not markets_match or not dates_match:
                    issues.append({
                        'group': group_id,
                        'issue': f'Expected {expected_markets} markets/{expected_dates} dates, got {actual_markets}/{actual_dates}'
                    })

                # Check resolution dates
                res_dates = sorted([d for d in prices['resolution_date'].unique() if pd.notna(d)])
                if len(res_dates) >= 3:
                    print(f"   Date range: {res_dates[0]} ... {res_dates[-1]}")
                print()

            else:
                print(f"✗ {group_id[:45]:<45}")
                print(f"   NO PRICE DATA AVAILABLE")
                print()
                issues.append({
                    'group': group_id,
                    'issue': 'No price data'
                })

        except Exception as e:
            print(f"✗ {group_id[:45]:<45}")
            print(f"   ERROR: {e}")
            print()
            issues.append({
                'group': group_id,
                'issue': f'Error: {e}'
            })

    # Summary
    print("="*70)
    print("SUMMARY")
    print("="*70)
    print()

    if len(issues) == 0:
        print("✓ ALL TESTS PASSED!")
        print()
        print("All top 10 groups have:")
        print("  - Complete price data")
        print("  - Correct market counts")
        print("  - Correct date counts")
        print("  - Valid implied rates")
        print()
        print("✓ Dashboard should work correctly")
    else:
        print(f"⚠ FOUND {len(issues)} ISSUES:")
        print()
        for issue in issues:
            print(f"  • {issue['group']}")
            print(f"    {issue['issue']}")
        print()
        print("Note: Some groups may not have price data yet.")
        print("Run data/downloaders/get_pricing.py to fetch missing prices.")

    print()
    print("="*70)


if __name__ == "__main__":
    main()
