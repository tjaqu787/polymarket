"""
Data loader for Polymarket modeling with term structure and implied rates.

This module loads market data, price history, and calculates:
- Implied rates from prices
- Term structure metrics (level, slope, curvature)
- Resolved outcomes for backtesting
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.implied_rates import (
    calculate_time_to_expiration,
    calculate_implied_rate
)
from utils.term_structure import (
    extract_term_structure,
    TermStructure
)


class PolymarketDataLoader:
    """
    Comprehensive data loader for Polymarket modeling.

    Loads market data from bets_for_timing_view with:
    - Market metadata (slugs, questions, resolution dates)
    - Price history
    - Implied rates
    - Term structure metrics
    - Resolved outcomes for backtesting
    """

    def __init__(self, db_path: str = "data/polymarket.db"):
        """
        Initialize the data loader.

        Args:
            db_path: Path to the SQLite database
        """
        self.db_path = db_path

    def get_market_data(self,
                       active_only: bool = False,
                       resolved_only: bool = False,
                       min_markets_per_group: int = 2) -> pd.DataFrame:
        """
        Load market data from bets_for_timing_view.

        Args:
            active_only: Only include active markets
            resolved_only: Only include resolved markets
            min_markets_per_group: Minimum number of markets per event group

        Returns:
            DataFrame with market metadata including slugs
        """
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT
                market_id,
                event_id AS market_group,
                token_id,
                outcome,
                token_index,
                question,
                market_slug,
                event_slug,
                event_title,
                resolution_date,
                end_date,
                closed_time,
                active,
                closed,
                archived,
                uma_resolution_status,
                outcome_prices_json,
                category,
                volume_num,
                liquidity_num
            FROM bets_for_timing_view
        """

        conditions = []
        if active_only:
            conditions.append("active = 1")
        if resolved_only:
            conditions.append("uma_resolution_status = 'resolved'")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        df = pd.read_sql_query(query, conn)
        conn.close()

        # Filter by minimum markets per group
        if min_markets_per_group > 1:
            market_counts = df.groupby('market_group')['market_id'].nunique()
            valid_groups = market_counts[market_counts >= min_markets_per_group].index
            df = df[df['market_group'].isin(valid_groups)]

        return df

    def get_price_data(self,
                      market_ids: Optional[List[str]] = None,
                      event_ids: Optional[List[str]] = None,
                      token_ids: Optional[List[str]] = None,
                      start_date: Optional[str] = None,
                      end_date: Optional[str] = None) -> pd.DataFrame:
        """
        Load price history data.

        Args:
            market_ids: Optional list of market IDs to filter
            event_ids: Optional list of event IDs to filter
            token_ids: Optional list of token IDs to filter
            start_date: Optional start date (YYYY-MM-DD)
            end_date: Optional end date (YYYY-MM-DD)

        Returns:
            DataFrame with price history
        """
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT
                ph.market_id,
                ph.event_id,
                ph.token_id,
                ph.outcome,
                ph.ts,
                ph.date,
                ph.price
            FROM price_history ph
        """

        conditions = []
        if market_ids:
            market_list = "', '".join(market_ids)
            conditions.append(f"ph.market_id IN ('{market_list}')")
        if event_ids:
            event_list = "', '".join(event_ids)
            conditions.append(f"ph.event_id IN ('{event_list}')")
        if token_ids:
            token_list = "', '".join(token_ids)
            conditions.append(f"ph.token_id IN ('{token_list}')")
        if start_date:
            conditions.append(f"ph.date >= '{start_date}'")
        if end_date:
            conditions.append(f"ph.date <= '{end_date}'")

        if conditions:
            query += " WHERE " + " AND ".join(conditions)

        query += " ORDER BY ph.event_id, ph.market_id, ph.outcome, ph.ts"

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df

    def get_resolved_outcomes(self, event_ids: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get resolved outcomes for backtesting.

        Args:
            event_ids: Optional list of event IDs to filter

        Returns:
            DataFrame with market_id, resolution_date, resolved_outcome, uma_resolution_status
        """
        conn = sqlite3.connect(self.db_path)

        query = """
            SELECT DISTINCT
                market_id,
                event_id AS market_group,
                market_slug,
                event_slug,
                question,
                resolution_date,
                closed_time,
                uma_resolution_status,
                outcome_prices_json,
                outcomes_json
            FROM bets_for_timing_view
            WHERE uma_resolution_status = 'resolved'
        """

        if event_ids:
            event_list = "', '".join(event_ids)
            query += f" AND event_id IN ('{event_list}')"

        df = pd.read_sql_query(query, conn)
        conn.close()

        # Parse outcome prices to determine which outcome won
        # outcome_prices_json has prices like ["0.999...", "0.000..."]
        # The outcome with price closest to 1.0 is typically the winner
        def parse_winner(row):
            """Parse the winning outcome from outcome_prices_json."""
            try:
                import json
                prices = json.loads(row['outcome_prices_json'])
                outcomes = json.loads(row['outcomes_json'])

                if len(prices) != len(outcomes):
                    return None

                # Find outcome with highest final price
                max_idx = np.argmax([float(p) for p in prices])
                return outcomes[max_idx]
            except:
                return None

        df['resolved_outcome'] = df.apply(parse_winner, axis=1)

        return df

    def calculate_implied_rates(self,
                               price_df: pd.DataFrame,
                               market_df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate implied rates for price data.

        Args:
            price_df: DataFrame with price history
            market_df: DataFrame with market metadata (must include resolution_date)

        Returns:
            DataFrame with additional columns: time_to_expiration, implied_rate
        """
        # Merge to get resolution dates
        df = price_df.merge(
            market_df[['token_id', 'resolution_date', 'event_slug', 'event_title', 'question']],
            on='token_id',
            how='left'
        )

        # Calculate time to expiration
        df['time_to_expiration'] = df.apply(
            lambda row: calculate_time_to_expiration(row['date'], row['resolution_date']),
            axis=1
        )

        # Calculate implied rate
        df['implied_rate'] = df.apply(
            lambda row: calculate_implied_rate(row['price'], row['time_to_expiration']),
            axis=1
        )

        # Also calculate complement implied rate
        df['complement_price'] = 1 - df['price']
        df['complement_implied_rate'] = df.apply(
            lambda row: calculate_implied_rate(row['complement_price'], row['time_to_expiration']),
            axis=1
        )

        return df

    def calculate_term_structure_metrics(self,
                                        rates_df: pd.DataFrame,
                                        outcome: str = 'Yes') -> pd.DataFrame:
        """
        Calculate term structure metrics for each date and event.

        Args:
            rates_df: DataFrame with implied rates (from calculate_implied_rates)
            outcome: Which outcome to calculate term structure for ('Yes' or 'No')

        Returns:
            DataFrame with date, market_group, and term structure metrics
        """
        metrics_list = []

        # Group by event and date
        for (event_id, date), group in rates_df.groupby(['event_id', 'date']):
            # Filter to specific outcome
            outcome_group = group[group['outcome'] == outcome].copy()

            if len(outcome_group) < 2:
                # Need at least 2 points for term structure
                continue

            # Remove NaN rates
            outcome_group = outcome_group.dropna(subset=['implied_rate', 'time_to_expiration'])

            if len(outcome_group) < 2:
                continue

            # Extract term structure
            maturities = outcome_group['time_to_expiration'].values
            rates = outcome_group['implied_rate'].values

            # Calculate metrics
            level = np.mean(rates)
            slope = rates[np.argmax(maturities)] - rates[np.argmin(maturities)] if len(rates) >= 2 else np.nan

            if len(rates) >= 3:
                # Curvature: (short + long) / 2 - medium
                sorted_idx = np.argsort(maturities)
                sorted_rates = rates[sorted_idx]
                mid_idx = len(sorted_rates) // 2
                curvature = (sorted_rates[0] + sorted_rates[-1]) / 2 - sorted_rates[mid_idx]
            else:
                curvature = np.nan

            metrics_list.append({
                'date': date,
                'market_group': event_id,
                'outcome': outcome,
                'ts_level': level,
                'ts_slope': slope,
                'ts_curvature': curvature,
                'ts_num_points': len(rates),
                'ts_min_maturity': np.min(maturities),
                'ts_max_maturity': np.max(maturities),
                'ts_maturity_spread': np.max(maturities) - np.min(maturities)
            })

        return pd.DataFrame(metrics_list)

    def load_full_dataset(self,
                         active_only: bool = False,
                         resolved_only: bool = False,
                         min_markets_per_group: int = 2,
                         outcome: str = 'Yes',
                         start_date: Optional[str] = None,
                         end_date: Optional[str] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Load complete dataset with prices, implied rates, and term structure metrics.

        Args:
            active_only: Only include active markets
            resolved_only: Only include resolved markets
            min_markets_per_group: Minimum markets per event group
            outcome: Outcome to calculate term structure for
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Tuple of (price_data_with_rates, term_structure_metrics, resolved_outcomes)
        """
        print("Loading market data...")
        market_df = self.get_market_data(
            active_only=active_only,
            resolved_only=resolved_only,
            min_markets_per_group=min_markets_per_group
        )
        print(f"Loaded {len(market_df)} markets in {market_df['market_group'].nunique()} event groups")

        print("\nLoading price data...")
        event_ids = market_df['market_group'].unique().tolist()
        price_df = self.get_price_data(
            event_ids=event_ids,
            start_date=start_date,
            end_date=end_date
        )
        print(f"Loaded {len(price_df)} price points")

        if len(price_df) == 0:
            print("No price data found!")
            return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

        print("\nCalculating implied rates...")
        rates_df = self.calculate_implied_rates(price_df, market_df)
        print(f"Calculated implied rates for {len(rates_df)} price points")

        print("\nCalculating term structure metrics...")
        ts_metrics_df = self.calculate_term_structure_metrics(rates_df, outcome=outcome)
        print(f"Calculated term structure metrics for {len(ts_metrics_df)} date-event combinations")

        # Merge term structure metrics back into rates_df
        if len(ts_metrics_df) > 0:
            rates_df = rates_df.merge(
                ts_metrics_df,
                on=['date', 'market_group'],
                how='left'
            )

        print("\nLoading resolved outcomes...")
        resolved_df = self.get_resolved_outcomes(event_ids=event_ids)
        print(f"Loaded {len(resolved_df)} resolved markets")

        return rates_df, ts_metrics_df, resolved_df


def main():
    """Example usage of the data loader."""
    loader = PolymarketDataLoader()

    print("="*60)
    print("POLYMARKET DATA LOADER - FULL DATASET")
    print("="*60)

    # Load full dataset
    rates_df, ts_metrics_df, resolved_df = loader.load_full_dataset(
        resolved_only=False,
        min_markets_per_group=3,
        outcome='Yes'
    )

    if len(rates_df) > 0:
        print("\n" + "="*60)
        print("SAMPLE DATA WITH TERM STRUCTURE")
        print("="*60)

        # Show sample of the data
        sample_cols = [
            'date', 'market_group', 'outcome', 'price',
            'time_to_expiration', 'implied_rate',
            'ts_level', 'ts_slope', 'ts_curvature', 'ts_num_points'
        ]
        available_cols = [col for col in sample_cols if col in rates_df.columns]
        print(rates_df[available_cols].head(20))

        print("\n" + "="*60)
        print("SUMMARY STATISTICS")
        print("="*60)
        print(f"Total price points: {len(rates_df)}")
        print(f"Date range: {rates_df['date'].min()} to {rates_df['date'].max()}")
        print(f"Event groups: {rates_df['market_group'].nunique()}")
        print(f"Markets: {rates_df['market_id'].nunique()}")

        print("\nImplied Rate Statistics:")
        print(rates_df['implied_rate'].describe())

        if 'ts_level' in rates_df.columns:
            print("\nTerm Structure Level Statistics:")
            print(rates_df['ts_level'].describe())

            print("\nTerm Structure Slope Statistics:")
            print(rates_df['ts_slope'].describe())

    if len(resolved_df) > 0:
        print("\n" + "="*60)
        print("RESOLVED OUTCOMES (for backtesting)")
        print("="*60)
        print(resolved_df[['market_id', 'question', 'resolution_date', 'resolved_outcome']].head(10))

        print(f"\nTotal resolved markets: {len(resolved_df)}")
        print(f"Outcomes distribution:")
        print(resolved_df['resolved_outcome'].value_counts())


if __name__ == "__main__":
    main()
