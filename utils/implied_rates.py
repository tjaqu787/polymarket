"""
Calculate implied hazard rates from Polymarket probabilities.

Models prediction market prices as cumulative probabilities in a Poisson process:
    P(T) = 1 - e^(-λT)

Where:
    P(T) = probability event occurs by time T (market price)
    T = time to expiration in years
    λ = hazard rate / intensity function (what we're solving for)

Solving for λ:
    λ = -ln(1 - P(T)) / T

This gives the implied rate at which the event is "happening" per unit time.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime


def calculate_time_to_expiration(current_date: str, resolution_date: str) -> float:
    """
    Calculate time to expiration in years.

    Args:
        current_date: ISO format date string (YYYY-MM-DD) or timestamp
        resolution_date: ISO format date string (YYYY-MM-DD)

    Returns:
        Time to expiration in years (float), or np.nan if resolution_date is None
    """
    # Handle None or missing resolution dates
    if resolution_date is None or pd.isna(resolution_date):
        return np.nan

    # Parse dates
    if 'T' in current_date:
        current_dt = datetime.fromisoformat(current_date.replace('Z', '+00:00'))
    else:
        current_dt = datetime.strptime(current_date, '%Y-%m-%d')

    resolution_dt = datetime.strptime(resolution_date, '%Y-%m-%d')

    # Calculate difference in days and convert to years
    days_to_expiration = (resolution_dt - current_dt).days
    years_to_expiration = days_to_expiration / 365.25

    return max(years_to_expiration, 1/365.25)  # Minimum 1 day to avoid division issues


def calculate_implied_rate(probability: float, time_to_expiration: float) -> float:
    """
    Calculate implied hazard rate from a prediction market price.

    Models the market price as P(T) = 1 - e^(-λT) where λ is the hazard rate.
    Solves for λ = -ln(1 - P(T)) / T

    Args:
        probability: Market probability (0 to 1) - probability event occurs by time T
        time_to_expiration: Time to expiration in years

    Returns:
        Implied hazard rate λ (annual rate at which event "happens")
    """
    if probability <= 0 or probability >= 1:
        return np.nan

    if time_to_expiration <= 0:
        return np.nan

    # λ = -ln(1 - P(T)) / T
    # This is the hazard rate for a Poisson process
    rate = -np.log(1 - probability) / time_to_expiration

    return rate


def load_price_history(db_path: str = "data/polymarket.db", market_group: str = None, use_semantic_groups: bool = True) -> pd.DataFrame:
    """
    Load price history for markets, optionally filtered by market_group.

    Args:
        db_path: Path to SQLite database
        market_group: Optional market_group (semantic_group_id or event_id) to filter by
        use_semantic_groups: Use semantic_group_id (default True) or event_id

    Returns:
        DataFrame with price history and market metadata
    """
    conn = sqlite3.connect(db_path)

    group_col = "semantic_group_id" if use_semantic_groups else "event_id"

    query = f"""
        SELECT
            ph.market_id,
            ph.event_id AS event_id,
            bv.semantic_group_id,
            bv.canonical_slug,
            bv.actor,
            ph.token_id,
            ph.outcome,
            ph.ts,
            ph.date,
            ph.price,
            bv.question,
            bv.event_slug,
            bv.event_title,
            bv.resolution_date,
            bv.end_date
        FROM price_history ph
        INNER JOIN bets_for_timing_view bv
            ON ph.token_id = bv.token_id
    """

    if market_group:
        query += f" WHERE bv.{group_col} = '{market_group}'"

    query += f" ORDER BY bv.{group_col}, ph.market_id, ph.outcome, ph.ts"

    df = pd.read_sql_query(query, conn)
    conn.close()

    # Add market_group column pointing to the appropriate grouping
    if use_semantic_groups:
        df['market_group'] = df['semantic_group_id']
    else:
        df['market_group'] = df['event_id']

    return df


def calculate_implied_rates_for_market_group(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate implied rates for all price points in a market group.

    Args:
        df: DataFrame from load_price_history

    Returns:
        DataFrame with additional columns for time_to_expiration and implied_rate
    """
    df = df.copy()

    # Calculate time to expiration for each price point
    df['time_to_expiration'] = [
        calculate_time_to_expiration(row['date'], row['resolution_date'])
        for _, row in df.iterrows()
    ]

    # Calculate implied rate
    df['implied_rate'] = [
        calculate_implied_rate(row['price'], row['time_to_expiration'])
        for _, row in df.iterrows()
    ]

    # Also calculate implied rate for the complement (1 - p)
    # This represents the rate implied by betting against the event
    df['complement_price'] = 1 - df['price']
    df['complement_implied_rate'] = [
        calculate_implied_rate(row['complement_price'], row['time_to_expiration'])
        for _, row in df.iterrows()
    ]

    return df


def get_market_groups(db_path: str = "data/polymarket.db", use_semantic_groups: bool = True) -> pd.DataFrame:
    """
    Get list of all market groups with metadata.

    Args:
        db_path: Path to database
        use_semantic_groups: Use semantic_group_id (default True) or event_id

    Returns:
        DataFrame with market_group, event_slug, event_title, and count of markets
    """
    conn = sqlite3.connect(db_path)

    group_col = "semantic_group_id" if use_semantic_groups else "market_group"

    query = f"""
        SELECT
            {group_col} as market_group,
            MAX(event_slug) as event_slug,
            MAX(event_title) as event_title,
            MAX(canonical_slug) as canonical_slug,
            MAX(actor) as actor,
            COUNT(DISTINCT market_id) AS num_markets,
            COUNT(DISTINCT resolution_date) AS num_dates,
            MIN(resolution_date) AS earliest_date,
            MAX(resolution_date) AS latest_date
        FROM bets_for_timing_view
        WHERE {group_col} IS NOT NULL
        GROUP BY {group_col}
        HAVING num_dates > 1
        ORDER BY num_dates DESC, num_markets DESC
    """

    df = pd.read_sql_query(query, conn)
    conn.close()

    return df


if __name__ == "__main__":
    # Example usage
    print("Getting market groups...")
    groups = get_market_groups()
    print(f"\nFound {len(groups)} market groups with multiple resolution dates")
    print("\nTop 10 market groups:")
    print(groups.head(10))

    # Example: Use first group that has price data
    # Try to find a group with price history
    conn = sqlite3.connect("data/polymarket.db")
    available_groups = pd.read_sql_query("""
        SELECT DISTINCT ph.event_id FROM price_history ph LIMIT 1
    """, conn)
    conn.close()

    if len(available_groups) > 0:
        example_group = available_groups.iloc[0]['event_id']
    else:
        example_group = '15042'  # Fallback to GPT-5

    print(f"\n\nCalculating implied rates for market group: {example_group}")

    df = load_price_history(market_group=example_group)
    print(f"Loaded {len(df)} price points")

    if len(df) > 0:
        df_with_rates = calculate_implied_rates_for_market_group(df)
        print("\nSample data with implied rates:")
        print(df_with_rates[['date', 'resolution_date', 'outcome', 'price',
                             'time_to_expiration', 'implied_rate']].head(10))
