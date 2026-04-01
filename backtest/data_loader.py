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
