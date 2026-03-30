"""
Build Semantic Market Groups

This script populates the semantic_market_groups table by:
1. Loading all markets from the database
2. Normalizing market slugs to remove timing patterns
3. Extracting actors/countries for splitting
4. Generating semantic group IDs
5. Writing results to database
6. Generating a review CSV for manual inspection

Usage:
    python3 data/build_semantic_groups.py
"""

import sqlite3
import pandas as pd
import sys
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent.parent))
from utils.slug_normalizer import normalize_and_group


def load_markets(db_path: str = "data/polymarket.db") -> pd.DataFrame:
    """
    Load markets that will be in bets_for_timing_view.

    We load from markets table with the same filters as timing_target_view.sql
    to ensure consistency.
    """
    conn = sqlite3.connect(db_path)

    query = """
        SELECT DISTINCT
            m.market_id,
            m.market_slug,
            m.question,
            e.slug AS event_slug
        FROM markets m
        INNER JOIN events e ON m.event_id = e.id
        WHERE (lower(m.question) LIKE '% by %'
            OR lower(m.question) LIKE '% before %'
            OR lower(m.question) LIKE '% no later than %'
            OR lower(m.question) LIKE '% until %')
        AND lower(m.question) NOT LIKE '% by more than %'
        AND lower(m.question) NOT LIKE '%nba%'
        AND lower(m.question) NOT LIKE '%nfl%'
        AND lower(m.question) NOT LIKE '%mlb%'
        AND lower(m.question) NOT LIKE '%all-time high%'
        AND lower(m.question) NOT LIKE '%points%'
        AND lower(m.question) NOT LIKE '% by at least %'
        AND lower(m.question) NOT LIKE '%eth%'
        AND lower(m.question) NOT LIKE '%$%'
        AND lower(m.question) NOT LIKE '%covid%'
        AND lower(m.question) NOT LIKE '%tweet %'
        AND lower(m.question) NOT LIKE '%market cap%'
        AND lower(m.question) NOT LIKE '%mcap%'
        AND lower(m.question) NOT LIKE '%usd%'
        AND lower(m.question) NOT LIKE '%candidate win%'
        AND lower(m.question) NOT LIKE '% win %'
        AND lower(m.question) NOT LIKE '%rcp%'
        AND lower(m.question) NOT LIKE '% case %'
        AND lower(m.question) NOT LIKE '% cases %'
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


def process_markets(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process markets to generate semantic grouping.

    For each market:
    - Normalize market_slug -> canonical_slug
    - Extract actor from slug/question
    - Generate semantic_group_id
    """
    results = []

    for _, row in df.iterrows():
        canonical_slug, actor, semantic_group_id = normalize_and_group(
            market_slug=row['market_slug'],
            question=row['question'],
            event_slug=row['event_slug']
        )

        results.append({
            'market_id': row['market_id'],
            'market_slug': row['market_slug'],
            'question': row['question'],
            'event_slug': row['event_slug'],
            'canonical_slug': canonical_slug,
            'actor': actor,
            'semantic_group_id': semantic_group_id
        })

    return pd.DataFrame(results)


def write_to_database(df: pd.DataFrame, db_path: str = "data/polymarket.db"):
    """
    Write semantic groups to database.

    Truncates existing data and inserts new records.
    """
    conn = sqlite3.connect(db_path)

    # Clear existing data
    conn.execute("DELETE FROM semantic_market_groups")

    # Insert new data
    df[['market_id', 'canonical_slug', 'actor', 'semantic_group_id']].to_sql(
        'semantic_market_groups',
        conn,
        if_exists='append',
        index=False
    )

    conn.commit()
    conn.close()


def generate_review_csv(df: pd.DataFrame, output_path: str = "data/semantic_groups_review.csv"):
    """
    Generate a review CSV for manual inspection.

    Groups markets by semantic_group_id and provides statistics:
    - Number of markets in group
    - Number of unique event_slugs
    - Number of unique actors
    - Sample questions
    - Flag for groups that need review
    """
    # Get some additional context from database
    conn = sqlite3.connect("data/polymarket.db")

    # Get resolution dates for each market
    dates_df = pd.read_sql_query("""
        SELECT DISTINCT
            m.market_id,
            SUBSTR(m.end_date, 1, 10) AS resolution_date
        FROM markets m
    """, conn)

    conn.close()

    # Merge with our data
    df_with_dates = df.merge(dates_df, on='market_id', how='left')

    # Group by semantic_group_id
    review_data = []

    for group_id in df_with_dates['semantic_group_id'].unique():
        group = df_with_dates[df_with_dates['semantic_group_id'] == group_id]

        num_markets = len(group)
        num_dates = group['resolution_date'].nunique()
        event_slugs = group['event_slug'].unique()
        actors = group['actor'].dropna().unique()
        sample_questions = group['question'].head(5).tolist()

        # Flag for review if:
        # - Mixed event_slugs (suggests false grouping)
        # - Very few markets (<2) or very many (>50)
        # - Mixed actors
        needs_review = (
            len(event_slugs) > 3 or  # Multiple different event_slugs
            num_markets < 2 or       # Too few markets
            num_markets > 50 or      # Too many markets
            len(actors) > 1          # Mixed actors
        )

        review_data.append({
            'semantic_group_id': group_id,
            'num_markets': num_markets,
            'num_dates': num_dates,
            'num_event_slugs': len(event_slugs),
            'event_slugs': ', '.join(event_slugs[:5]),  # First 5
            'actors': ', '.join(actors) if len(actors) > 0 else None,
            'sample_questions': ' | '.join(sample_questions),
            'needs_review': needs_review
        })

    review_df = pd.DataFrame(review_data)

    # Sort by needs_review first, then by num_markets descending
    review_df = review_df.sort_values(['needs_review', 'num_markets'], ascending=[False, False])

    # Write to CSV
    review_df.to_csv(output_path, index=False)

    return review_df


def main():
    """Main execution function."""
    print("="*70)
    print("BUILDING SEMANTIC MARKET GROUPS")
    print("="*70)
    print()

    # Step 1: Load markets
    print("Step 1: Loading markets from database...")
    markets_df = load_markets()
    print(f"  Loaded {len(markets_df):,} markets")
    print()

    # Step 2: Process markets
    print("Step 2: Processing markets (normalizing slugs, extracting actors)...")
    processed_df = process_markets(markets_df)
    print(f"  Processed {len(processed_df):,} markets")
    print(f"  Found {processed_df['semantic_group_id'].nunique():,} semantic groups")
    print()

    # Show sample
    print("Sample results:")
    print(processed_df[['market_slug', 'canonical_slug', 'actor', 'semantic_group_id']].head(10))
    print()

    # Step 3: Write to database
    print("Step 3: Writing to database...")
    write_to_database(processed_df)
    print("  ✓ Written to semantic_market_groups table")
    print()

    # Step 4: Generate review CSV
    print("Step 4: Generating review CSV...")
    review_df = generate_review_csv(processed_df)
    print(f"  ✓ Written to data/semantic_groups_review.csv")
    print(f"  Total groups: {len(review_df):,}")
    print(f"  Groups needing review: {review_df['needs_review'].sum():,}")
    print()

    # Show statistics
    print("="*70)
    print("STATISTICS")
    print("="*70)
    print(f"Total markets: {len(processed_df):,}")
    print(f"Total semantic groups: {processed_df['semantic_group_id'].nunique():,}")
    print(f"Groups with time series (≥2 dates): {len(review_df[review_df['num_dates'] >= 2]):,}")
    print(f"Groups with rich time series (≥5 dates): {len(review_df[review_df['num_dates'] >= 5]):,}")
    print()

    # Show top groups by market count
    print("Top 10 largest semantic groups:")
    print(review_df[['semantic_group_id', 'num_markets', 'num_dates']].head(10).to_string(index=False))
    print()

    print("✓ Done! Review data/semantic_groups_review.csv for quality check.")


if __name__ == "__main__":
    main()
