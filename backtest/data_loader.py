"""
Data loading utilities for backtesting engine.
"""

import sqlite3
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict
import numpy as np


class DataLoader:
    """Loads and prepares historical Polymarket data for backtesting."""

    def __init__(self, db_path: str):
        """
        Initialize DataLoader.

        Args:
            db_path: Path to polymarket.db SQLite database
        """
        self.db_path = db_path

    def load_market_data(
        self,
        market_ids: Optional[List[str]] = None,
        event_ids: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_volume: float = 0,
        include_inactive: bool = False
    ) -> pd.DataFrame:
        """
        Load market price history data.

        Args:
            market_ids: Specific market IDs to load (optional)
            event_ids: Specific event IDs to load (optional)
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            min_volume: Minimum market volume filter
            include_inactive: Include inactive markets

        Returns:
            DataFrame with columns: market_id, event_id, token_id, outcome,
                                   date, price, question, resolution_date, etc.
        """
        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT
            m.market_id,
            m.event_id,
            m.question,
            m.end_date,
            m.volume_num,
            m.liquidity_num,
            m.active,
            m.closed,
            mt.token_id,
            mt.outcome,
            ph.ts,
            ph.date,
            ph.price,
            e.title as event_title,
            e.slug as event_slug,
            DATE(m.end_date) as resolution_date
        FROM markets m
        JOIN market_tokens mt ON m.market_id = mt.market_id
        JOIN price_history ph ON mt.token_id = ph.token_id
        JOIN events e ON m.event_id = e.id
        WHERE 1=1
        """

        params = []

        if market_ids:
            placeholders = ','.join('?' * len(market_ids))
            query += f" AND m.market_id IN ({placeholders})"
            params.extend(market_ids)

        if event_ids:
            placeholders = ','.join('?' * len(event_ids))
            query += f" AND m.event_id IN ({placeholders})"
            params.extend(event_ids)

        if start_date:
            query += " AND ph.date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND ph.date <= ?"
            params.append(end_date)

        if min_volume > 0:
            query += " AND m.volume_num >= ?"
            params.append(min_volume)

        if not include_inactive:
            query += " AND m.active = 1"

        query += " ORDER BY ph.date, m.market_id, mt.outcome"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        # Convert date columns to datetime
        df['date'] = pd.to_datetime(df['date'])
        df['resolution_date'] = pd.to_datetime(df['resolution_date'])

        # Calculate time to expiration
        df['time_to_expiration'] = (df['resolution_date'] - df['date']).dt.days / 365.25
        df['time_to_expiration'] = df['time_to_expiration'].clip(lower=1/365.25)  # Min 1 day

        # Calculate implied rate for both Yes and No outcomes
        df['implied_rate'] = np.nan

        # Handle Yes outcomes
        yes_mask = df['outcome'] == 'Yes'
        if yes_mask.any():
            yes_price_clipped = df.loc[yes_mask, 'price'].clip(1e-6, 1-1e-6)
            df.loc[yes_mask, 'implied_rate'] = -np.log(yes_price_clipped) / df.loc[yes_mask, 'time_to_expiration']

        # Handle No outcomes (convert to Yes first)
        no_mask = df['outcome'] == 'No'
        if no_mask.any():
            yes_price = 1 - df.loc[no_mask, 'price']
            yes_price_clipped = yes_price.clip(1e-6, 1-1e-6)
            df.loc[no_mask, 'implied_rate'] = -np.log(yes_price_clipped) / df.loc[no_mask, 'time_to_expiration']

        return df

    def load_timing_markets(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        use_semantic_groups: bool = False
    ) -> pd.DataFrame:
        """
        Load timing-based prediction markets (uses bets_for_timing_view).

        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            min_volume: Minimum market volume

        Returns:
            DataFrame with timing market data
        """
        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT
            bft.market_id,
            bft.market_group,
            bft.event_id,
            bft.semantic_group_id,
            bft.canonical_slug,
            bft.actor,
            bft.question,
            bft.event_title,
            bft.event_slug,
            bft.resolution_date,
            bft.end_date,
            bft.volume_num,
            bft.liquidity_num,
            bft.token_id,
            bft.outcome,
            ph.date,
            ph.ts,
            ph.price
        FROM bets_for_timing_view bft
        JOIN price_history ph 
        ON bft.token_id = ph.token_id
        WHERE 1=1
        """

        params = []

        if start_date:
            query += " AND ph.date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND ph.date <= ?"
            params.append(end_date)

        query += " ORDER BY ph.date, bft.market_group, bft.resolution_date"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        # Convert dates
        df['date'] = pd.to_datetime(df['date'])
        df['resolution_date'] = pd.to_datetime(df['resolution_date'])

        # Create grouping column based on parameter
        if use_semantic_groups and 'semantic_group_id' in df.columns and df['semantic_group_id'].notna().any():
            df['group_col'] = df['semantic_group_id'].fillna(df['event_id'])
        else:
            df['group_col'] = df['event_id']  # Fallback to event_id

        # Calculate metrics
        df['time_to_expiration'] = (df['resolution_date'] - df['date']).dt.days / 365.25
        df['time_to_expiration'] = df['time_to_expiration'].clip(lower=1/365.25)

        # Calculate implied rate for No outcomes (convert to Yes first for consistency)
        df['implied_rate'] = np.nan
        no_mask = df['outcome'] == 'No'
        if no_mask.any():
            yes_price = 1 - df.loc[no_mask, 'price']
            # Clip to avoid log(0) or log(negative)
            yes_price_clipped = yes_price.clip(1e-6, 1-1e-6)
            df.loc[no_mask, 'implied_rate'] = -np.log(yes_price_clipped) / df.loc[no_mask, 'time_to_expiration']

        # Also handle Yes outcomes if they exist (for backward compatibility)
        yes_mask = df['outcome'] == 'Yes'
        if yes_mask.any():
            yes_price_clipped = df.loc[yes_mask, 'price'].clip(1e-6, 1-1e-6)
            df.loc[yes_mask, 'implied_rate'] = -np.log(yes_price_clipped) / df.loc[yes_mask, 'time_to_expiration']

        return df

    def get_market_groups(self) -> pd.DataFrame:
        """
        Get all market groups (events with multiple resolution dates).

        Returns:
            DataFrame with market group metadata
        """
        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT
            market_group,
            event_slug,
            event_title,
            COUNT(DISTINCT market_id) as num_markets,
            COUNT(DISTINCT resolution_date) as num_dates,
            MIN(resolution_date) as earliest_resolution,
            MAX(resolution_date) as latest_resolution
        FROM bets_for_timing_view
        GROUP BY market_group
        HAVING num_dates > 1
        ORDER BY num_markets DESC
        """

        df = pd.read_sql_query(query, conn)
        conn.close()

        return df

    def load_term_structure_data(
        self,
        market_group: str,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Load data for a specific market group for term structure analysis.

        Args:
            market_group: Event ID (market_group)
            start_date: Start date filter
            end_date: End date filter

        Returns:
            DataFrame with term structure data
        """
        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT
            bft.*,
            ph.date,
            ph.ts,
            ph.price
        FROM bets_for_timing_view bft
        JOIN price_history ph ON bft.token_id = ph.token_id
        WHERE bft.market_group = ?
        """

        params = [market_group]

        if start_date:
            query += " AND ph.date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND ph.date <= ?"
            params.append(end_date)

        query += " ORDER BY ph.date, bft.resolution_date"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        # Convert and calculate
        df['date'] = pd.to_datetime(df['date'])
        df['resolution_date'] = pd.to_datetime(df['resolution_date'])
        df['time_to_expiration'] = (df['resolution_date'] - df['date']).dt.days / 365.25
        df['time_to_expiration'] = df['time_to_expiration'].clip(lower=1/365.25)

        # Calculate implied rate for both Yes and No outcomes
        df['implied_rate'] = np.nan

        # Handle Yes outcomes
        yes_mask = df['outcome'] == 'Yes'
        if yes_mask.any():
            yes_price_clipped = df.loc[yes_mask, 'price'].clip(1e-6, 1-1e-6)
            df.loc[yes_mask, 'implied_rate'] = -np.log(yes_price_clipped) / df.loc[yes_mask, 'time_to_expiration']

        # Handle No outcomes (convert to Yes first)
        no_mask = df['outcome'] == 'No'
        if no_mask.any():
            yes_price = 1 - df.loc[no_mask, 'price']
            yes_price_clipped = yes_price.clip(1e-6, 1-1e-6)
            df.loc[no_mask, 'implied_rate'] = -np.log(yes_price_clipped) / df.loc[no_mask, 'time_to_expiration']

        return df

    def load_carry_markets(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        min_volume: float = 0
    ) -> pd.DataFrame:
        """
        Load markets for carry trading strategy.

        Unlike timing markets, this loads ALL active markets with semantic grouping,
        allowing the strategy to filter based on TTE and probability thresholds.

        Args:
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            min_volume: Minimum market volume

        Returns:
            DataFrame with market data including semantic groups
        """
        conn = sqlite3.connect(self.db_path)

        query = """
        SELECT
            m.market_id,
            m.event_id,
            COALESCE(smg.semantic_group_id, m.event_id) as group_col,
            smg.semantic_group_id,
            smg.canonical_slug,
            smg.actor,
            m.question,
            m.category,
            e.slug AS event_slug,
            e.title AS event_title,
            DATE(m.end_date) AS resolution_date,
            m.end_date,
            m.volume_num,
            m.liquidity_num,
            m.active,
            m.closed,
            mt.token_id,
            mt.outcome,
            ph.date,
            ph.ts,
            ph.price
        FROM markets m
        INNER JOIN market_tokens mt ON m.market_id = mt.market_id
        INNER JOIN price_history ph ON mt.token_id = ph.token_id
        INNER JOIN events e ON m.event_id = e.id
        LEFT JOIN semantic_market_groups smg ON m.market_id = smg.market_id
        WHERE 1=1
        """

        params = []

        if start_date:
            query += " AND ph.date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND ph.date <= ?"
            params.append(end_date)

        if min_volume > 0:
            query += " AND m.volume_num >= ?"
            params.append(min_volume)

        # Don't filter by active - markets can become inactive after expiry
        # but we still need their historical data for backtesting
        query += " ORDER BY ph.date, m.market_id, mt.outcome"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        # Convert dates
        df['date'] = pd.to_datetime(df['date'])
        df['resolution_date'] = pd.to_datetime(df['resolution_date'])

        # Calculate time to expiration
        df['time_to_expiration'] = (df['resolution_date'] - df['date']).dt.days / 365.25
        df['time_to_expiration'] = df['time_to_expiration'].clip(lower=1/365.25)

        # Calculate implied rate for both Yes and No outcomes
        df['implied_rate'] = np.nan

        # Handle Yes outcomes
        yes_mask = df['outcome'] == 'Yes'
        if yes_mask.any():
            yes_price_clipped = df.loc[yes_mask, 'price'].clip(1e-6, 1-1e-6)
            df.loc[yes_mask, 'implied_rate'] = -np.log(yes_price_clipped) / df.loc[yes_mask, 'time_to_expiration']

        # Handle No outcomes (convert to Yes first)
        no_mask = df['outcome'] == 'No'
        if no_mask.any():
            yes_price = 1 - df.loc[no_mask, 'price']
            yes_price_clipped = yes_price.clip(1e-6, 1-1e-6)
            df.loc[no_mask, 'implied_rate'] = -np.log(yes_price_clipped) / df.loc[no_mask, 'time_to_expiration']

        return df

    def get_available_dates(
        self,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ) -> List[str]:
        """
        Get all dates with available price data.

        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Sorted list of dates (YYYY-MM-DD strings)
        """
        conn = sqlite3.connect(self.db_path)

        query = "SELECT DISTINCT date FROM price_history WHERE 1=1"
        params = []

        if start_date:
            query += " AND date >= ?"
            params.append(start_date)

        if end_date:
            query += " AND date <= ?"
            params.append(end_date)

        query += " ORDER BY date"

        df = pd.read_sql_query(query, conn, params=params)
        conn.close()

        return df['date'].tolist()

    def load_resolved_events_for_eb(
        self,
        holdout_end_date: str,
        lookback_days: int = 30
    ) -> pd.DataFrame:
        """
        Load markets for Empirical Bayes factor fitting.

        Treats all markets with end_date <= holdout_end_date as "resolved" for training,
        and extracts term structure snapshots from `lookback_days` before their resolution.

        Args:
            holdout_end_date: Only use markets with end_date <= this date (treated as resolved)
            lookback_days: Days before resolution to take term structure snapshot (default 30)

        Returns:
            DataFrame with columns:
                - event_id or semantic_group_id
                - category
                - resolution_date
                - date (observation date for term structure)
                - times (array of tenors)
                - cdf_values (array of Yes prices)
                - ts_slope, ts_curvature, implied_rate
        """
        conn = sqlite3.connect(self.db_path)

        # Get markets with end_date in training period (treat as resolved even if not technically closed)
        query = """
        SELECT
            m.market_id,
            m.event_id,
            COALESCE(smg.semantic_group_id, m.event_id) as group_col,
            smg.semantic_group_id,
            m.question,
            m.category,
            m.end_date as resolution_date
        FROM markets m
        LEFT JOIN semantic_market_groups smg ON m.market_id = smg.market_id
        WHERE DATE(m.end_date) <= ?
          AND m.end_date IS NOT NULL
          AND m.category IS NOT NULL
          AND m.volume_num >= 100
        """

        params = [holdout_end_date]

        df_markets = pd.read_sql_query(query, conn, params=params)
        conn.close()

        if df_markets.empty:
            print(f"No markets found with end_date <= {holdout_end_date}")
            return pd.DataFrame()

        print(f"Found {len(df_markets)} markets with end_date <= {holdout_end_date}")

        # For each semantic group with multiple markets, extract term structure
        # snapshot from `lookback_days` before the earliest resolution
        resolved_events = []

        for group_id in df_markets['group_col'].unique():
            group_markets = df_markets[df_markets['group_col'] == group_id]

            # Need at least 3 markets in a group to fit a term structure
            if len(group_markets) < 3:
                continue

            # Get earliest resolution date in group
            resolution_dates = pd.to_datetime(group_markets['resolution_date'])
            earliest_resolution = resolution_dates.min()

            # Calculate snapshot date (lookback_days before earliest resolution)
            snapshot_date = earliest_resolution - timedelta(days=lookback_days)
            snapshot_date_str = snapshot_date.strftime('%Y-%m-%d')

            # Load price data for this group at the snapshot date
            market_ids = group_markets['market_id'].tolist()

            try:
                # Query price data near the snapshot date (±3 days tolerance)
                conn = sqlite3.connect(self.db_path)

                placeholders = ','.join('?' * len(market_ids))
                price_query = f"""
                SELECT
                    mt.market_id,
                    mt.outcome,
                    ph.date,
                    ph.price,
                    m.end_date as resolution_date
                FROM market_tokens mt
                JOIN price_history ph ON mt.token_id = ph.token_id
                JOIN markets m ON mt.market_id = m.market_id
                WHERE mt.market_id IN ({placeholders})
                  AND mt.outcome = 'No'
                  AND ph.date >= DATE(?, '-3 days')
                  AND ph.date <= DATE(?, '+3 days')
                ORDER BY mt.market_id, ph.date
                """

                params_price = market_ids + [snapshot_date_str, snapshot_date_str]
                df_prices = pd.read_sql_query(price_query, conn, params=params_price)
                conn.close()

                if df_prices.empty:
                    continue

                # Convert dates
                df_prices['date'] = pd.to_datetime(df_prices['date'])
                df_prices['resolution_date'] = pd.to_datetime(df_prices['resolution_date'])

                # Take the closest date to snapshot date for each market
                df_prices['date_diff'] = (df_prices['date'] - snapshot_date).abs()
                df_snapshot = df_prices.sort_values('date_diff').groupby('market_id').first().reset_index()

                if len(df_snapshot) < 3:
                    continue

                # Use the actual snapshot date from the data
                actual_snapshot_date = df_snapshot['date'].iloc[0]

                # Calculate times and CDF values
                times = []
                cdf_values = []
                implied_rates = []

                for _, row in df_snapshot.iterrows():
                    time_to_resolution = (row['resolution_date'] - actual_snapshot_date).days / 365.25
                    if time_to_resolution <= 0:
                        continue

                    # Convert No price to Yes price
                    yes_price = 1.0 - row['price']
                    yes_price_clipped = np.clip(yes_price, 1e-6, 1-1e-6)

                    # Calculate implied rate
                    implied_rate = -np.log(yes_price_clipped) / time_to_resolution

                    times.append(time_to_resolution)
                    cdf_values.append(yes_price)
                    implied_rates.append(implied_rate)

                if len(times) < 3:
                    continue

                # Calculate term structure features
                times_arr = np.array(times)
                cdf_arr = np.array(cdf_values)
                rates_arr = np.array(implied_rates)

                # Sort by time
                sort_idx = np.argsort(times_arr)
                times_arr = times_arr[sort_idx]
                cdf_arr = cdf_arr[sort_idx]
                rates_arr = rates_arr[sort_idx]

                # Calculate slope (change in rate over time)
                if len(times_arr) >= 2:
                    ts_slope = (rates_arr[-1] - rates_arr[0]) / (times_arr[-1] - times_arr[0])
                else:
                    ts_slope = 0.0

                # Calculate curvature (second derivative approximation)
                if len(times_arr) >= 3:
                    # Use central difference for middle points
                    mid_idx = len(rates_arr) // 2
                    if mid_idx > 0 and mid_idx < len(rates_arr) - 1:
                        h1 = times_arr[mid_idx] - times_arr[mid_idx-1]
                        h2 = times_arr[mid_idx+1] - times_arr[mid_idx]
                        d1 = (rates_arr[mid_idx] - rates_arr[mid_idx-1]) / h1
                        d2 = (rates_arr[mid_idx+1] - rates_arr[mid_idx]) / h2
                        ts_curvature = (d2 - d1) / ((h1 + h2) / 2)
                    else:
                        ts_curvature = 0.0
                else:
                    ts_curvature = 0.0

                # Average implied rate
                avg_implied_rate = np.mean(rates_arr)

                # Get category from first market in group
                category = group_markets.iloc[0]['category']
                if pd.isna(category):
                    category = 'unknown'

                resolved_events.append({
                    'semantic_group_id': group_id,
                    'event_id': group_markets.iloc[0]['event_id'],
                    'category': category,
                    'resolution_date': earliest_resolution,
                    'date': actual_snapshot_date,
                    'times': times_arr,
                    'cdf_values': cdf_arr,
                    'ts_slope': ts_slope,
                    'ts_curvature': ts_curvature,
                    'implied_rate': avg_implied_rate,
                    'n_markets': len(times_arr)
                })

            except Exception as e:
                # Skip groups that fail
                continue

        if len(resolved_events) == 0:
            print("No resolved events with valid term structures found")
            return pd.DataFrame()

        df_resolved = pd.DataFrame(resolved_events)
        print(f"Successfully extracted {len(df_resolved)} resolved event term structures")
        print(f"Categories: {df_resolved['category'].value_counts().to_dict()}")

        return df_resolved
