#!/usr/bin/env python3
"""
Quick arbitrage analysis using sampling.
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

conn = sqlite3.connect(DB_PATH)

# Sample a few specific dates to speed up analysis
print("Sampling data from specific dates...")
sample_dates = ['2025-12-01', '2026-01-01', '2026-02-01', '2026-03-01']

query = f"""
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
    AND p.date IN ('{"','".join(sample_dates)}')
ORDER BY p.event_id, p.date
"""

df = pd.read_sql_query(query, conn)
conn.close()

print(f"Loaded {len(df)} records from {df['event_id'].nunique()} events")

# Extract target dates
df['target_date'] = df['question'].apply(extract_target_date)
df = df[df['target_date'].notna()].copy()
df['date'] = pd.to_datetime(df['date'])
df['days_to_target'] = (df['target_date'] - df['date']).dt.days
df = df[df['days_to_target'] > 0].copy()

print(f"Analyzing {len(df)} price points...")

inversions = []

for (event_id, date), group in df.groupby(['event_id', 'date']):
    if len(group) < 2:
        continue

    group = group.sort_values('days_to_target')

    for i in range(len(group)):
        for j in range(i + 1, len(group)):
            row_i = group.iloc[i]
            row_j = group.iloc[j]

            if row_j['price'] < row_i['price']:
                spread = row_i['price'] - row_j['price']
                inversions.append({
                    'event_id': event_id,
                    'date': date,
                    'short_days': row_i['days_to_target'],
                    'short_price': row_i['price'],
                    'short_question': row_i['question'],
                    'long_days': row_j['days_to_target'],
                    'long_price': row_j['price'],
                    'long_question': row_j['question'],
                    'spread': spread,
                    'spread_pct': (spread / row_i['price']) * 100
                })

inversions_df = pd.DataFrame(inversions)

print(f"\n{'='*80}")
print(f"RESULTS (Sample of {len(sample_dates)} dates)")
print(f"{'='*80}")
print(f"Total inversions: {len(inversions_df):,}")
print(f"Events with inversions: {inversions_df['event_id'].nunique()}")
print(f"Inversion rate: {len(inversions_df) / max(1, len(df)) * 100:.2f}% of observations")

if len(inversions_df) > 0:
    print(f"\nSpread statistics:")
    print(f"  Mean: {inversions_df['spread'].mean():.4f} ({inversions_df['spread_pct'].mean():.2f}%)")
    print(f"  Median: {inversions_df['spread'].median():.4f} ({inversions_df['spread_pct'].median():.2f}%)")
    print(f"  Max: {inversions_df['spread'].max():.4f} ({inversions_df['spread_pct'].max():.2f}%)")

    print(f"\nTop 5 largest spreads:")
    for idx, row in inversions_df.nlargest(5, 'spread').iterrows():
        print(f"  {row['spread']:.4f} ({row['spread_pct']:.1f}%) - {row['short_days']}d @ {row['short_price']:.3f} vs {row['long_days']}d @ {row['long_price']:.3f}")
