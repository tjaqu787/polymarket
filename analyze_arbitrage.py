#!/usr/bin/env python3
"""
Analyze non-monotonic price structures in time-distributed events.
Find arbitrage opportunities where longer-dated contracts < shorter-dated contracts.
"""

import sqlite3
import pandas as pd
import re
from datetime import datetime
from collections import defaultdict

DB_PATH = "data/polymarket.db"

def extract_target_date(question: str):
    """Extract target date from question text."""
    patterns = [
        r'by ([A-Za-z]+) (\d+), (\d{4})',
        r'before ([A-Za-z]+) (\d+), (\d{4})',
    ]
    for pattern in patterns:
        match = re.search(pattern, question)
        if match:
            try:
                month_name, day, year = match.groups()
                date_str = f"{year}-{month_name}-{day}"
                return datetime.strptime(date_str, '%Y-%B-%d')
            except:
                continue
    return None

# Load timing markets data
conn = sqlite3.connect(DB_PATH)

# Get all timing markets (questions with "by" in them)
query = """
SELECT
    p.event_id,
    p.market_id,
    m.question,
    p.date,
    p.outcome,
    p.price
FROM price_history p
JOIN markets m ON p.market_id = m.market_id
WHERE m.question LIKE '% by %'
    AND p.outcome = 'Yes'
    AND p.date BETWEEN '2025-11-05' AND '2026-03-16'
ORDER BY p.event_id, p.date, m.question
"""

print("Loading timing markets data...")
df = pd.read_sql_query(query, conn)
conn.close()

print(f"Loaded {len(df)} records from {df['event_id'].nunique()} events")

# Extract target dates
print("\nExtracting target dates from questions...")
df['target_date'] = df['question'].apply(extract_target_date)

# Filter to only rows where we successfully extracted a target date
df = df[df['target_date'].notna()].copy()
print(f"Found {len(df)} records with valid target dates")

# Convert to days from observation date
df['date'] = pd.to_datetime(df['date'])
df['days_to_target'] = (df['target_date'] - df['date']).dt.days

# Filter to future targets only
df = df[df['days_to_target'] > 0].copy()
print(f"Filtered to {len(df)} records with future target dates")

# Find inversions (non-monotonic structures)
print("\nAnalyzing price structures for inversions...")

inversions = []
event_inversion_counts = defaultdict(lambda: {'dates': set(), 'total_inversions': 0})

# Group by event and date
for (event_id, date), group in df.groupby(['event_id', 'date']):
    # Need at least 2 time buckets to have an inversion
    if len(group) < 2:
        continue

    # Sort by days to target
    group = group.sort_values('days_to_target')

    # Check for inversions
    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            row_i = group.iloc[i]
            row_j = group.iloc[j]

            # row_j has more days to target than row_i
            # Price should be higher (or equal) for longer dated
            if row_j['price'] < row_i['price']:
                # Inversion found!
                spread = row_i['price'] - row_j['price']
                inversions.append({
                    'event_id': event_id,
                    'date': date,
                    'short_market': row_i['market_id'],
                    'short_question': row_i['question'],
                    'short_days': row_i['days_to_target'],
                    'short_price': row_i['price'],
                    'long_market': row_j['market_id'],
                    'long_question': row_j['question'],
                    'long_days': row_j['days_to_target'],
                    'long_price': row_j['price'],
                    'spread': spread,
                    'spread_pct': (spread / row_i['price']) * 100
                })

                event_inversion_counts[event_id]['dates'].add(date)
                event_inversion_counts[event_id]['total_inversions'] += 1

inversions_df = pd.DataFrame(inversions)

if len(inversions_df) > 0:
    print(f"\n{'='*80}")
    print(f"INVERSION ANALYSIS RESULTS")
    print(f"{'='*80}")
    print(f"Total inversions found: {len(inversions_df):,}")
    print(f"Events with inversions: {len(event_inversion_counts)}")
    print(f"Trading days with inversions: {inversions_df['date'].nunique()}")

    print(f"\n{'='*80}")
    print(f"SPREAD STATISTICS")
    print(f"{'='*80}")
    print(f"Mean spread: {inversions_df['spread'].mean():.4f} ({inversions_df['spread_pct'].mean():.2f}%)")
    print(f"Median spread: {inversions_df['spread'].median():.4f} ({inversions_df['spread_pct'].median():.2f}%)")
    print(f"Max spread: {inversions_df['spread'].max():.4f} ({inversions_df['spread_pct'].max():.2f}%)")
    print(f"Min spread: {inversions_df['spread'].min():.4f} ({inversions_df['spread_pct'].min():.2f}%)")

    print(f"\n{'='*80}")
    print(f"TOP 10 LARGEST SPREADS")
    print(f"{'='*80}")
    top_spreads = inversions_df.nlargest(10, 'spread')
    for idx, row in top_spreads.iterrows():
        print(f"\nDate: {row['date'].strftime('%Y-%m-%d')}")
        print(f"  SHORT: {row['short_question'][:60]}...")
        print(f"         {row['short_days']} days @ {row['short_price']:.4f}")
        print(f"  LONG:  {row['long_question'][:60]}...")
        print(f"         {row['long_days']} days @ {row['long_price']:.4f}")
        print(f"  SPREAD: {row['spread']:.4f} ({row['spread_pct']:.2f}%)")

    print(f"\n{'='*80}")
    print(f"EVENTS WITH MOST INVERSIONS")
    print(f"{'='*80}")
    top_events = sorted(event_inversion_counts.items(),
                       key=lambda x: x[1]['total_inversions'],
                       reverse=True)[:10]

    for event_id, data in top_events:
        print(f"\nEvent: {event_id}")
        print(f"  Total inversions: {data['total_inversions']}")
        print(f"  Trading days affected: {len(data['dates'])}")

        # Get sample question
        sample = df[df['event_id'] == event_id].iloc[0]
        print(f"  Sample: {sample['question'][:70]}...")

    # Save results
    inversions_df.to_csv('arbitrage_opportunities.csv', index=False)
    print(f"\n{'='*80}")
    print(f"Full results saved to: arbitrage_opportunities.csv")
    print(f"{'='*80}")

else:
    print("\nNo inversions found - all price structures are monotonic!")
