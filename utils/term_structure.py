"""
Term structure analysis of implied rates from Polymarket.

This module analyzes how implied rates vary across different time horizons,
similar to yield curve analysis in fixed income markets.
"""

import sqlite3
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Tuple, Dict
from scipy.optimize import minimize
from scipy.interpolate import UnivariateSpline

from utils.implied_rates import (
    load_price_history,
    calculate_implied_rates_for_market_group,
    get_market_groups
)


class TermStructure:
    """
    Represents the term structure of implied rates at a specific point in time.
    """

    def __init__(self, date: str, maturities: np.ndarray, rates: np.ndarray,
                 event_id: str = None, event_title: str = None):
        """
        Initialize term structure.

        Args:
            date: Observation date (YYYY-MM-DD)
            maturities: Array of times to maturity (in years)
            rates: Array of implied rates corresponding to maturities
            event_id: Optional event identifier
            event_title: Optional event description
        """
        self.date = date
        self.maturities = maturities
        self.rates = rates
        self.event_id = event_id
        self.event_title = event_title

        # Sort by maturity
        sort_idx = np.argsort(self.maturities)
        self.maturities = self.maturities[sort_idx]
        self.rates = self.rates[sort_idx]

    def slope(self) -> float:
        """
        Calculate the slope of the term structure (long rate - short rate).
        Uses the longest and shortest available maturities.
        """
        if len(self.rates) < 2:
            return np.nan
        return self.rates[-1] - self.rates[0]

    def level(self) -> float:
        """Calculate average rate across all maturities."""
        return np.mean(self.rates)

    def curvature(self) -> float:
        """
        Calculate curvature: (short + long) / 2 - medium.
        Uses simple approximation with available data points.
        """
        if len(self.rates) < 3:
            return np.nan

        # Use first, middle, and last points
        mid_idx = len(self.rates) // 2
        return (self.rates[0] + self.rates[-1]) / 2 - self.rates[mid_idx]

    def interpolate(self, target_maturities: np.ndarray, method='spline') -> np.ndarray:
        """
        Interpolate rates at target maturities.

        Args:
            target_maturities: Array of maturities to interpolate at
            method: 'linear' or 'spline'

        Returns:
            Array of interpolated rates
        """
        if method == 'spline':
            # Use cubic spline with smoothing
            spline = UnivariateSpline(self.maturities, self.rates, k=min(3, len(self.rates)-1), s=0.1)
            return spline(target_maturities)
        else:
            # Linear interpolation
            return np.interp(target_maturities, self.maturities, self.rates)


def extract_term_structure(df: pd.DataFrame, observation_date: str,
                           outcome_filter: str = 'Yes') -> Optional[TermStructure]:
    """
    Extract term structure for a specific observation date.

    Args:
        df: DataFrame with implied rates (from calculate_implied_rates_for_market_group)
        observation_date: Date to extract term structure for (YYYY-MM-DD)
        outcome_filter: Which outcome to use ('Yes' or 'No')

    Returns:
        TermStructure object or None if no data available
    """
    # Filter to the specific date and outcome
    df_date = df[(df['date'] == observation_date) & (df['outcome'] == outcome_filter)].copy()

    if len(df_date) == 0:
        return None

    # Remove NaN rates
    df_date = df_date.dropna(subset=['implied_rate', 'time_to_expiration'])

    if len(df_date) == 0:
        return None

    # Extract maturities and rates
    maturities = df_date['time_to_expiration'].values
    rates = df_date['implied_rate'].values

    # Get event metadata
    event_id = df_date['market_group'].iloc[0] if 'market_group' in df_date.columns else None
    event_title = df_date['event_title'].iloc[0] if 'event_title' in df_date.columns else None

    return TermStructure(observation_date, maturities, rates, event_id, event_title)


def extract_term_structure_history(df: pd.DataFrame, outcome_filter: str = 'Yes') -> Dict[str, TermStructure]:
    """
    Extract term structure for all available dates.

    Args:
        df: DataFrame with implied rates
        outcome_filter: Which outcome to use ('Yes' or 'No')

    Returns:
        Dictionary mapping date -> TermStructure
    """
    term_structures = {}

    for date in sorted(df['date'].unique()):
        ts = extract_term_structure(df, date, outcome_filter)
        if ts is not None and len(ts.maturities) > 0:
            term_structures[date] = ts

    return term_structures


def calculate_term_structure_metrics(term_structures: Dict[str, TermStructure]) -> pd.DataFrame:
    """
    Calculate term structure metrics over time.

    Args:
        term_structures: Dictionary of date -> TermStructure

    Returns:
        DataFrame with metrics: date, level, slope, curvature
    """
    metrics = []

    for date, ts in sorted(term_structures.items()):
        metrics.append({
            'date': date,
            'level': ts.level(),
            'slope': ts.slope(),
            'curvature': ts.curvature(),
            'num_points': len(ts.maturities),
            'min_maturity': ts.maturities.min() if len(ts.maturities) > 0 else np.nan,
            'max_maturity': ts.maturities.max() if len(ts.maturities) > 0 else np.nan,
        })

    return pd.DataFrame(metrics)


def nelson_siegel(tau: np.ndarray, beta0: float, beta1: float, beta2: float, lambda_: float) -> np.ndarray:
    """
    Nelson-Siegel yield curve model.

    r(tau) = beta0 + beta1 * ((1 - exp(-lambda*tau)) / (lambda*tau))
             + beta2 * (((1 - exp(-lambda*tau)) / (lambda*tau)) - exp(-lambda*tau))

    Args:
        tau: Time to maturity (years)
        beta0: Level parameter
        beta1: Slope parameter
        beta2: Curvature parameter
        lambda_: Decay parameter

    Returns:
        Array of fitted rates
    """
    # Avoid division by zero
    tau = np.maximum(tau, 1e-6)

    term1 = (1 - np.exp(-lambda_ * tau)) / (lambda_ * tau)
    term2 = term1 - np.exp(-lambda_ * tau)

    return beta0 + beta1 * term1 + beta2 * term2


def fit_nelson_siegel(ts: TermStructure) -> Tuple[Dict[str, float], np.ndarray]:
    """
    Fit Nelson-Siegel model to term structure.

    Args:
        ts: TermStructure object

    Returns:
        Tuple of (parameters dict, fitted rates)
    """
    def objective(params):
        beta0, beta1, beta2, lambda_ = params
        fitted = nelson_siegel(ts.maturities, beta0, beta1, beta2, lambda_)
        return np.sum((fitted - ts.rates) ** 2)

    # Initial guess
    x0 = [ts.level(), ts.slope(), 0.01, 1.0]

    # Bounds for parameters
    bounds = [
        (None, None),      # beta0
        (None, None),      # beta1
        (None, None),      # beta2
        (0.01, 10.0),      # lambda (must be positive)
    ]

    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

    if result.success:
        beta0, beta1, beta2, lambda_ = result.x
        fitted_rates = nelson_siegel(ts.maturities, beta0, beta1, beta2, lambda_)

        params = {
            'beta0': beta0,
            'beta1': beta1,
            'beta2': beta2,
            'lambda': lambda_,
            'rmse': np.sqrt(np.mean((fitted_rates - ts.rates) ** 2))
        }

        return params, fitted_rates
    else:
        return None, None


def analyze_event_term_structure(event_id: str, db_path: str = "data/polymarket.db",
                                 outcome: str = 'Yes') -> Dict:
    """
    Complete term structure analysis for a single event.

    Args:
        event_id: Event ID to analyze
        db_path: Path to database
        outcome: Outcome to analyze

    Returns:
        Dictionary with analysis results
    """
    print(f"Loading data for event {event_id}...")
    df = load_price_history(db_path, market_group=event_id)

    if len(df) == 0:
        print(f"No price data found for event {event_id}")
        return None

    print(f"Loaded {len(df)} price points")

    # Calculate implied rates
    print("Calculating implied rates...")
    df_rates = calculate_implied_rates_for_market_group(df)

    # Extract term structures
    print("Extracting term structures...")
    term_structures = extract_term_structure_history(df_rates, outcome_filter=outcome)
    print(f"Found {len(term_structures)} term structures")

    if len(term_structures) == 0:
        print("No valid term structures found")
        return None

    # Calculate metrics
    print("Calculating term structure metrics...")
    metrics = calculate_term_structure_metrics(term_structures)

    # Get latest term structure
    latest_date = sorted(term_structures.keys())[-1]
    latest_ts = term_structures[latest_date]

    results = {
        'event_id': event_id,
        'event_title': latest_ts.event_title,
        'num_dates': len(term_structures),
        'date_range': (min(term_structures.keys()), max(term_structures.keys())),
        'latest_term_structure': latest_ts,
        'term_structures': term_structures,
        'metrics': metrics,
        'df_rates': df_rates
    }

    return results


if __name__ == "__main__":
    print("Term Structure Analysis\n" + "="*50)

    # Find an event with sufficient data
    conn = sqlite3.connect("data/polymarket.db")
    query = """
        SELECT
            ph.event_id,
            COUNT(DISTINCT ph.date) as num_dates,
            COUNT(DISTINCT ph.market_id) as num_markets,
            MIN(bv.event_title) as event_title
        FROM price_history ph
        INNER JOIN bets_for_timing_view bv ON ph.token_id = bv.token_id
        GROUP BY ph.event_id
        HAVING num_dates >= 5 AND num_markets >= 3
        ORDER BY num_dates DESC, num_markets DESC
        LIMIT 10
    """

    events_df = pd.read_sql_query(query, conn)
    conn.close()

    print("\nEvents with sufficient data for term structure analysis:")
    print(events_df)

    if len(events_df) > 0:
        # Analyze the first event
        event_id = events_df.iloc[0]['event_id']
        print(f"\n\nAnalyzing event: {event_id}")
        print(f"Title: {events_df.iloc[0]['event_title']}")

        results = analyze_event_term_structure(event_id)

        if results:
            print("\n" + "="*50)
            print("SUMMARY STATISTICS")
            print("="*50)
            print(f"Event: {results['event_title']}")
            print(f"Number of observations: {results['num_dates']}")
            print(f"Date range: {results['date_range'][0]} to {results['date_range'][1]}")

            latest_ts = results['latest_term_structure']
            print(f"\nLatest Term Structure ({latest_ts.date}):")
            print(f"  Level: {latest_ts.level()*100:.2f}%")
            print(f"  Slope: {latest_ts.slope()*100:.2f}%")
            print(f"  Curvature: {latest_ts.curvature()*100:.2f}%")

            print("\nMetrics summary:")
            print(results['metrics'].describe())
    else:
        print("\nNo events found with sufficient data for analysis.")
        print("Run get_pricing.py first to fetch price history data.")
