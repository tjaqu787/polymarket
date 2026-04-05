"""
Survival Conditional Strategy

Trades conditional (survival-adjusted) probability mispricing in time-distributed prediction markets.

Markets quote P(event by T). As time passes without resolution, correct pricing requires
conditioning on survival:
    P(event in [T1, T2] | not yet) = [F(T2) - F(T1)] / [1 - F(T1)]

Markets are slow to reprice this. We exploit the gap.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict, Tuple
from datetime import datetime, timedelta

from backtest.strategy import Strategy, Signal, SignalType


class SurvivalConditionalStrategy(Strategy):
    """
    Trades conditional (survival-adjusted) probability mispricing.

    Exploits the fact that markets are slow to update P(event by T) as time passes
    without the event occurring. The correct conditional probability should be:
        P(event in [T1, T2] | survived to T1) = [F(T2) - F(T1)] / S(T1)
    where S(T1) = 1 - F(T1) is the survival function.
    """

    def __init__(
        self,
        config: Dict,
        **kwargs
    ):
        """
        Initialize Survival Conditional Strategy.

        Args:
            config: Configuration dictionary with keys:
                - db_path: Path to database
                - min_tte_days: Min days to expiry (default 7)
                - max_tte_days: Max days to expiry (default 365)
                - refit_days: Days between model refits (default 7)
                - min_survival_edge: Min edge to trade (default 0.04)
                - min_volume: Min market volume (default 500)
                - kelly_fraction: Fractional Kelly (default 0.25)
                - max_position: Max position per contract (default 0.10)
                - max_group_exposure: Max exposure per event group (default 0.15)
                - distribution: CDF model (default 'gamma')
                - roll_forward_days: Days between edge re-evaluation (default 7)
        """
        super().__init__()

        # Configuration
        self.db_path = config.get('db_path')
        self.min_tte_days = config.get('min_tte_days', 7)
        self.max_tte_days = config.get('max_tte_days', 365)
        self.refit_days = config.get('refit_days', 7)
        self.min_survival_edge = config.get('min_survival_edge', 0.04)
        self.min_volume = config.get('min_volume', 500)
        self.kelly_fraction = config.get('kelly_fraction', 0.25)
        self.max_position = config.get('max_position', 0.10)
        self.max_group_exposure = config.get('max_group_exposure', 0.15)
        self.distribution = config.get('distribution', 'gamma')
        self.roll_forward_days = config.get('roll_forward_days', 7)

        # State tracking
        self.fitted_models = {}  # event_id -> (fit_date, cdf_function)
        self.last_roll_forward = {}  # (market_id, outcome) -> last_check_date
        self.position_entry_dates = {}  # (market_id, outcome) -> entry_date

        # Statistics
        self.opportunities_found = 0
        self.trades_executed = 0
        self.edges_collapsed = 0
        self.rolled_forward = 0

        # Load data loader
        from backtest.data_loader import DataLoader
        self.data_loader = DataLoader(self.db_path)

    def name(self) -> str:
        return f"SurvivalConditional_edge{int(self.min_survival_edge*100)}_kelly{int(self.kelly_fraction*100)}"

    def _fit_cdf_for_event(
        self,
        event_id: str,
        event_data: pd.DataFrame,
        current_date: pd.Timestamp
    ) -> Optional[callable]:
        """
        Fit a CDF model to the term structure for an event group.

        Returns:
            A callable F(t) that returns P(event by time t) or None if fit fails
        """
        try:
            # Import the Gamma fitter
            from models.factored_gamma_model.gamma_fitter import GammaCDFFitter

            # Extract term structure: time to expiry vs Yes price
            term_structure = []
            for _, row in event_data.iterrows():
                if row['outcome'] == 'Yes':
                    tte_days = (row['resolution_date'] - current_date).days
                    if tte_days > 0:
                        term_structure.append({
                            'time_to_expiry': tte_days / 365.25,  # Convert to years
                            'yes_price': row['price']
                        })

            if len(term_structure) < 2:
                return None

            df_term = pd.DataFrame(term_structure)

            # Fit Gamma CDF
            fitter = GammaCDFFitter()
            success, params = fitter.fit(
                times=df_term['time_to_expiry'].values,
                cumulative_probs=df_term['yes_price'].values
            )

            if not success:
                return None

            # Create CDF function F(t)
            def cdf_func(t_years):
                """Return P(event by time t)"""
                return fitter.cdf(t_years, params['shape'], params['rate'])

            return cdf_func

        except Exception as e:
            return None

    def _compute_conditional_probability(
        self,
        cdf_func: callable,
        t1_years: float,
        t2_years: float
    ) -> Optional[float]:
        """
        Compute P(event in [T1, T2] | survived to T1).

        Args:
            cdf_func: CDF function F(t)
            t1_years: Current time (survival point)
            t2_years: Contract expiry time

        Returns:
            Conditional probability or None if invalid
        """
        if t1_years >= t2_years:
            return None

        try:
            F_t1 = cdf_func(t1_years)
            F_t2 = cdf_func(t2_years)
            S_t1 = 1 - F_t1

            # Check for numerical instability
            if S_t1 < 0.05:  # Event 95%+ likely already happened
                return None

            if F_t2 <= F_t1:  # CDF should be monotonic
                return None

            conditional_prob = (F_t2 - F_t1) / S_t1

            # Clip to valid probability range
            conditional_prob = np.clip(conditional_prob, 0.0, 1.0)

            return conditional_prob

        except Exception:
            return None

    def _should_refit(self, event_id: str, current_date: pd.Timestamp) -> bool:
        """Check if we need to refit the model for this event."""
        if event_id not in self.fitted_models:
            return True

        last_fit_date = self.fitted_models[event_id][0]
        days_since_fit = (current_date - last_fit_date).days

        return days_since_fit >= self.refit_days

    def _should_roll_forward(
        self,
        market_id: str,
        outcome: str,
        current_date: pd.Timestamp
    ) -> bool:
        """Check if we should re-evaluate an open position."""
        key = (market_id, outcome)

        if key not in self.last_roll_forward:
            return True

        last_check = self.last_roll_forward[key]
        days_since_check = (current_date - last_check).days

        return days_since_check >= self.roll_forward_days

    def on_data(
        self,
        current_date: pd.Timestamp,
        data: pd.DataFrame,
        portfolio_value: float = 10000.0,
        **kwargs
    ) -> List[Signal]:
        """
        Generate survival conditional trading signals.

        Args:
            current_date: Current date
            data: Market data
            portfolio_value: Current portfolio value

        Returns:
            List of trading signals
        """
        signals = []

        # Group by event_id
        group_col = 'event_id'
        if group_col not in data.columns:
            return signals

        for event_id in data[group_col].unique():
            if pd.isna(event_id):
                continue

            event_data = data[data[group_col] == event_id].copy()

            # Filter for active markets with sufficient volume
            event_data = event_data[
                (event_data['volume_num'] >= self.min_volume) if 'volume_num' in event_data.columns else True
            ]

            if event_data.empty:
                continue

            # Calculate time to expiry
            event_data['days_to_expiry'] = (
                event_data['resolution_date'] - current_date
            ).dt.days

            # Filter by TTE range
            event_data = event_data[
                (event_data['days_to_expiry'] >= self.min_tte_days) &
                (event_data['days_to_expiry'] <= self.max_tte_days)
            ]

            if len(event_data) < 2:
                continue

            # Refit CDF if needed
            if self._should_refit(event_id, current_date):
                cdf_func = self._fit_cdf_for_event(event_id, event_data, current_date)
                if cdf_func is not None:
                    self.fitted_models[event_id] = (current_date, cdf_func)

            # Skip if no fitted model
            if event_id not in self.fitted_models:
                continue

            _, cdf_func = self.fitted_models[event_id]

            # Check each contract for mispricing
            for _, row in event_data.iterrows():
                if row['outcome'] != 'Yes':
                    continue

                market_id = row['market_id']
                market_price = row['price']
                t2_years = row['days_to_expiry'] / 365.25

                # T1 is current time (0 years from now)
                t1_years = 0.0

                # Compute conditional fair value
                conditional_fair = self._compute_conditional_probability(
                    cdf_func, t1_years, t2_years
                )

                if conditional_fair is None:
                    continue

                # Compute edge
                edge = market_price - conditional_fair

                # Check if we have a position
                has_position = self.portfolio and (market_id, 'Yes') in self.portfolio.positions

                if has_position:
                    # Roll forward check: should we exit?
                    if self._should_roll_forward(market_id, 'Yes', current_date):
                        self.last_roll_forward[(market_id, 'Yes')] = current_date
                        self.rolled_forward += 1

                        # Exit if edge collapsed or contract too close to expiry
                        if abs(edge) < self.min_survival_edge / 2 or row['days_to_expiry'] < self.min_tte_days:
                            signals.append(Signal(
                                market_id=market_id,
                                token_id=row['token_id'],
                                outcome='Yes',
                                signal_type=SignalType.CLOSE,
                                size=0,  # Close all
                                price=market_price,
                                reason=f"Edge collapsed or near expiry: edge={edge:.4f}",
                                metadata={'strategy_type': 'survival_conditional'}
                            ))
                            self.edges_collapsed += 1

                else:
                    # Entry signal: trade if edge exceeds threshold
                    if abs(edge) >= self.min_survival_edge:
                        self.opportunities_found += 1

                        # Determine direction
                        if edge > 0:
                            # Market overprices -> SHORT (sell Yes / buy No)
                            signal_type = SignalType.SHORT
                            outcome = 'Yes'
                            reason = f"Survival edge: market={market_price:.3f} > fair={conditional_fair:.3f}, edge={edge:.4f}"
                        else:
                            # Market underprices -> LONG (buy Yes)
                            signal_type = SignalType.BUY
                            outcome = 'Yes'
                            reason = f"Survival edge: market={market_price:.3f} < fair={conditional_fair:.3f}, edge={edge:.4f}"

                        # Position sizing using fractional Kelly
                        kelly_f = abs(edge) / (edge ** 2 + 0.01)  # Simple variance estimate
                        position_fraction = self.kelly_fraction * kelly_f
                        position_fraction = min(position_fraction, self.max_position)

                        position_size = (portfolio_value * position_fraction) / market_price
                        position_size = round(position_size)

                        if position_size > 0:
                            signals.append(Signal(
                                market_id=market_id,
                                token_id=row['token_id'],
                                outcome=outcome,
                                signal_type=signal_type,
                                size=position_size,
                                price=market_price,
                                reason=reason,
                                metadata={
                                    'strategy_type': 'survival_conditional',
                                    'edge': edge,
                                    'conditional_fair': conditional_fair,
                                    'event_id': event_id
                                }
                            ))

                            self.trades_executed += 1
                            self.position_entry_dates[(market_id, outcome)] = current_date
                            self.last_roll_forward[(market_id, outcome)] = current_date

        return signals

    def on_backtest_end(self, **kwargs):
        """Print strategy statistics."""
        print(f"\n{'='*70}")
        print("SURVIVAL CONDITIONAL STRATEGY STATISTICS")
        print(f"{'='*70}")
        print(f"Opportunities found:             {self.opportunities_found}")
        print(f"Trades executed:                 {self.trades_executed}")
        print(f"Edges collapsed (exits):         {self.edges_collapsed}")
        print(f"Roll forward checks:             {self.rolled_forward}")
        print(f"\nStrategy Parameters:")
        print(f"  Min survival edge:             {self.min_survival_edge*100:.1f}%")
        print(f"  Kelly fraction:                {self.kelly_fraction*100:.1f}%")
        print(f"  Max position:                  {self.max_position*100:.1f}%")
        print(f"  Refit interval:                {self.refit_days} days")
        print(f"  Roll forward interval:         {self.roll_forward_days} days")
        print(f"  Distribution:                  {self.distribution}")
        print(f"{'='*70}\n")
