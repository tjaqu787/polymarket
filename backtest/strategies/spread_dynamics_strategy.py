"""
SpreadDynamicsStrategy - Trades the anticipated change in implied rate spreads.

This strategy operates on the velocity and direction of spread change between
adjacent contracts in term structures, conditioned on volume regime.

Unlike strategies that react to current mispricings, this one positions before
the spread moves, by detecting regime conditions that historically precede
spread compression or widening.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from backtest.strategy import Strategy, Signal, SignalType
from utils.implied_rates import calculate_implied_rate, calculate_time_to_expiration


class SpreadDynamicsStrategy(Strategy):
    """
    Trades the anticipated change in implied rate spreads, not the level.

    Enters positions before expected spread compression or widening,
    using volume regime, spread velocity, and rate momentum as signals.

    Two trade types:
        - COMPRESSION trade: short the wide leg, long the tight leg
          (spread expected to narrow after news/volume spike)
        - WIDENING trade: long the wide leg, short the tight leg
          (spread expected to widen during inactivity)
    """

    def __init__(self, config: Optional[Dict] = None):
        super().__init__(config)

        # Strategy parameters
        self.lookback_days = self.config.get('lookback_days', 7)
        self.vol_lookback_days = self.config.get('vol_lookback_days', 3)
        self.vol_spike_threshold = self.config.get('vol_spike_threshold', 2.0)
        self.vol_drought_threshold = self.config.get('vol_drought_threshold', -0.5)
        self.min_spread_change = self.config.get('min_spread_change', 0.05)
        self.min_spread_level = self.config.get('min_spread_level', 0.02)
        self.min_tte_days = self.config.get('min_tte_days', 14)
        self.min_volume = self.config.get('min_volume', 300)
        self.refit_days = self.config.get('refit_days', 1)
        self.kelly_fraction = self.config.get('kelly_fraction', 0.25)
        self.max_position = self.config.get('max_position', 0.10)
        self.max_event_exposure = self.config.get('max_event_exposure', 0.20)
        self.min_pair_correlation = self.config.get('min_pair_correlation', 0.60)

        # State tracking
        self.spread_positions = {}  # (event_id, near_market_id, far_market_id) -> trade info
        self.spread_history = {}  # pair_id -> DataFrame with spread time series
        self.volume_history = {}  # event_id -> DataFrame with volume time series
        self.last_refit_date = None
        self.spread_velocity_history = {}  # pair_id -> list of velocities for variance estimation

    @property
    def name(self) -> str:
        return (f"SpreadDynamics_LB{self.lookback_days}_"
                f"Vol{self.vol_lookback_days}_"
                f"MinSpd{int(self.min_spread_change*100)}")

    def calculate_volume_zscore(self, event_id: str, current_date: pd.Timestamp,
                                data: pd.DataFrame) -> float:
        """
        Calculate volume z-score for regime classification.

        Args:
            event_id: Event group ID (not used, data is pre-filtered)
            current_date: Current date
            data: Historical data (pre-filtered for this event)

        Returns:
            Volume z-score (standardized volume change)
        """
        # Data is already filtered for this event, no need to filter again
        if data.empty:
            return 0.0

        # OPTIMIZATION: Vectorized volume aggregation
        daily_volume = data.groupby('date')['volume_num'].sum()

        if len(daily_volume) < self.vol_lookback_days + 1:
            return 0.0

        # Get recent volume history (vectorized filtering)
        lookback_end = current_date - pd.Timedelta(days=1)
        lookback_start = lookback_end - pd.Timedelta(days=self.vol_lookback_days)

        recent_volumes = daily_volume[(daily_volume.index >= lookback_start) &
                                      (daily_volume.index <= lookback_end)]

        if len(recent_volumes) < 2:
            return 0.0

        # Vectorized z-score calculation
        current_volume = daily_volume.get(current_date, 0)
        mean_volume = recent_volumes.mean()
        std_volume = recent_volumes.std()

        if std_volume == 0 or pd.isna(std_volume):
            return 0.0

        z_score = (current_volume - mean_volume) / std_volume
        return z_score

    def build_spread_pairs(self, event_id: str, current_date: pd.Timestamp,
                          data: pd.DataFrame) -> List[Tuple[Dict, Dict]]:
        """
        Build adjacent pairs of contracts for spread trading.

        Args:
            event_id: Event group ID (not used, data is pre-filtered)
            current_date: Current date
            data: Market data (pre-filtered for this event)

        Returns:
            List of (near_contract, far_contract) tuples
        """
        pairs = []

        # Data is already filtered for this event, just filter by date and outcome
        event_data = data[
            (data['date'] == current_date) &
            (data['outcome'] == 'No')
        ]

        if event_data.empty:
            return pairs

        # Calculate days to expiration
        event_data['days_to_expiry'] = (
            event_data['resolution_date'] - current_date
        ).dt.days

        # Filter by minimum TTE and volume
        event_data = event_data[
            (event_data['days_to_expiry'] >= self.min_tte_days) &
            (event_data['volume_num'] >= self.min_volume)
        ]

        if len(event_data) < 2:
            return pairs

        # Sort by time to expiration
        event_data = event_data.sort_values('days_to_expiry')

        # Create adjacent pairs
        for i in range(len(event_data) - 1):
            near = event_data.iloc[i]
            far = event_data.iloc[i + 1]

            near_dict = {
                'market_id': near['market_id'],
                'token_id': near['token_id'],
                'price': near['price'],
                'tte_years': near['time_to_expiration'],
                'tte_days': near['days_to_expiry'],
                'volume': near['volume_num']
            }

            far_dict = {
                'market_id': far['market_id'],
                'token_id': far['token_id'],
                'price': far['price'],
                'tte_years': far['time_to_expiration'],
                'tte_days': far['days_to_expiry'],
                'volume': far['volume_num']
            }

            pairs.append((near_dict, far_dict))

        # Also add the full-span pair if we have more than 2 contracts
        if len(event_data) > 2:
            near = event_data.iloc[0]
            far = event_data.iloc[-1]

            near_dict = {
                'market_id': near['market_id'],
                'token_id': near['token_id'],
                'price': near['price'],
                'tte_years': near['time_to_expiration'],
                'tte_days': near['days_to_expiry'],
                'volume': near['volume_num']
            }

            far_dict = {
                'market_id': far['market_id'],
                'token_id': far['token_id'],
                'price': far['price'],
                'tte_years': far['time_to_expiration'],
                'tte_days': far['days_to_expiry'],
                'volume': far['volume_num']
            }

            pairs.append((near_dict, far_dict))

        return pairs

    def calculate_spread(self, near: Dict, far: Dict) -> Optional[float]:
        """
        Calculate implied rate spread between two contracts.

        Args:
            near: Near contract data
            far: Far contract data

        Returns:
            Spread (λ_far - λ_near) or None if calculation fails
        """
        # Calculate implied rates
        lambda_near = calculate_implied_rate(near['price'], near['tte_years'])
        lambda_far = calculate_implied_rate(far['price'], far['tte_years'])

        if pd.isna(lambda_near) or pd.isna(lambda_far):
            return None

        spread = lambda_far - lambda_near
        return spread

    def get_spread_history(self, pair_id: str, current_date: pd.Timestamp,
                          data: pd.DataFrame, near_market_id: str,
                          far_market_id: str) -> Optional[pd.DataFrame]:
        """
        Get or compute spread history for a pair.

        Args:
            pair_id: Unique pair identifier
            current_date: Current date
            data: Historical data (pre-filtered for this event)
            near_market_id: Near contract market ID
            far_market_id: Far contract market ID

        Returns:
            DataFrame with dates and spread values, or None if insufficient data
        """
        # OPTIMIZATION: Check cache first
        if pair_id in self.spread_history:
            cached = self.spread_history[pair_id]
            # If cache has current_date, return it
            if current_date in cached['date'].values:
                return cached[cached['date'] <= current_date]

        # Get historical data for both markets (data is already event-filtered)
        near_data = data[
            (data['market_id'] == near_market_id) &
            (data['outcome'] == 'No') &
            (data['date'] <= current_date)
        ].sort_values('date')

        far_data = data[
            (data['market_id'] == far_market_id) &
            (data['outcome'] == 'No') &
            (data['date'] <= current_date)
        ].sort_values('date')

        if near_data.empty or far_data.empty:
            return None

        # Merge on date
        merged = pd.merge(
            near_data[['date', 'price', 'time_to_expiration']],
            far_data[['date', 'price', 'time_to_expiration']],
            on='date',
            suffixes=('_near', '_far')
        )

        if len(merged) < self.lookback_days:
            return None

        # OPTIMIZATION: Vectorized spread calculation instead of iterrows
        # Convert to numpy arrays for faster computation
        price_near = merged['price_near'].values
        price_far = merged['price_far'].values
        tte_near = merged['time_to_expiration_near'].values
        tte_far = merged['time_to_expiration_far'].values

        # Clip prices to avoid log(0) or log(negative)
        yes_price_near = np.clip(1 - price_near, 1e-6, 1-1e-6)
        yes_price_far = np.clip(1 - price_far, 1e-6, 1-1e-6)

        # Vectorized implied rate calculation: λ = -ln(1-p) / t
        lambda_near = -np.log(yes_price_near) / tte_near
        lambda_far = -np.log(yes_price_far) / tte_far

        # Calculate spreads
        spreads = lambda_far - lambda_near
        merged['spread'] = spreads
        merged = merged.dropna(subset=['spread'])

        result = merged[['date', 'spread']]

        # OPTIMIZATION: Cache the result (keep last 30 days to limit memory)
        if len(result) > 0:
            self.spread_history[pair_id] = result.tail(30)

        return result

    def generate_spread_signals(self, event_id: str, current_date: pd.Timestamp,
                               data: pd.DataFrame) -> List[Signal]:
        """
        Generate signals for spread trades.

        Args:
            event_id: Event group ID
            current_date: Current date
            data: Historical data

        Returns:
            List of trading signals
        """
        signals = []

        # Calculate volume regime
        vol_zscore = self.calculate_volume_zscore(event_id, current_date, data)

        # Classify regime
        if vol_zscore > self.vol_spike_threshold:
            volume_regime = 'news'  # Expect compression
        elif vol_zscore < self.vol_drought_threshold:
            volume_regime = 'inactive'  # Expect widening
        else:
            volume_regime = 'neutral'  # No clear signal

        # Build spread pairs
        pairs = self.build_spread_pairs(event_id, current_date, data)

        for near, far in pairs:
            pair_id = f"{event_id}_{near['market_id']}_{far['market_id']}"
            position_key = (event_id, near['market_id'], far['market_id'])

            # Calculate current spread
            current_spread = self.calculate_spread(near, far)
            if current_spread is None or abs(current_spread) < self.min_spread_level:
                continue

            # Get spread history
            spread_hist = self.get_spread_history(
                pair_id, current_date, data,
                near['market_id'], far['market_id']
            )

            if spread_hist is None or len(spread_hist) < self.lookback_days:
                continue

            # Calculate spread velocity
            spread_lookback = spread_hist.iloc[-self.lookback_days]['spread']
            spread_change = current_spread - spread_lookback
            spread_velocity = spread_change / self.lookback_days

            # Check if we already have a position
            existing_position = self.spread_positions.get(position_key)

            # === EXIT SIGNALS ===
            if existing_position:
                entry_spread = existing_position['entry_spread']
                trade_type = existing_position['trade_type']

                # Check exit conditions
                spread_moved = current_spread - entry_spread

                # Take profit: spread moved in expected direction by ≥ 50% of min_spread_change
                if trade_type == 'compression' and spread_moved < -0.5 * self.min_spread_change:
                    # Spread compressed as expected - close both legs
                    signals.extend(self._generate_close_signals(
                        position_key, near, far, current_spread,
                        f"Take profit: Spread compressed {spread_moved:.4f}"
                    ))
                    continue

                elif trade_type == 'widening' and spread_moved > 0.5 * self.min_spread_change:
                    # Spread widened as expected - close both legs
                    signals.extend(self._generate_close_signals(
                        position_key, near, far, current_spread,
                        f"Take profit: Spread widened {spread_moved:.4f}"
                    ))
                    continue

                # Stop loss: spread moved against position
                if trade_type == 'compression' and spread_moved > self.min_spread_change:
                    signals.extend(self._generate_close_signals(
                        position_key, near, far, current_spread,
                        f"Stop loss: Spread widened {spread_moved:.4f}"
                    ))
                    continue

                elif trade_type == 'widening' and spread_moved < -self.min_spread_change:
                    signals.extend(self._generate_close_signals(
                        position_key, near, far, current_spread,
                        f"Stop loss: Spread compressed {spread_moved:.4f}"
                    ))
                    continue

                # Volume regime flip
                entry_regime = 'news' if existing_position['entry_vol_zscore'] > self.vol_spike_threshold else 'inactive'
                if entry_regime != volume_regime and volume_regime != 'neutral':
                    signals.extend(self._generate_close_signals(
                        position_key, near, far, current_spread,
                        f"Regime flip: {entry_regime} -> {volume_regime}"
                    ))
                    continue

                # TTE check
                if near['tte_days'] < self.min_tte_days or far['tte_days'] < self.min_tte_days:
                    signals.extend(self._generate_close_signals(
                        position_key, near, far, current_spread,
                        "TTE below minimum"
                    ))
                    continue

            # === ENTRY SIGNALS ===
            else:
                # Skip if change is too small
                if abs(spread_change) < self.min_spread_change:
                    continue

                # Compression trade: spread widening + volume spike
                if (spread_velocity > 0 and volume_regime == 'news'):
                    # Position size based on edge
                    position_size = self._calculate_position_size(
                        pair_id, abs(spread_velocity)
                    )

                    if position_size > 0:
                        # SHORT far leg, LONG near leg
                        signals.append(Signal(
                            market_id=far['market_id'],
                            token_id=far['token_id'],
                            outcome='No',
                            signal_type=SignalType.SELL,
                            size=position_size,
                            price=far['price'],
                            reason=f"Spread compression: vol_spike (z={vol_zscore:.2f}) + spread_widening (Δ={spread_change:.4f})",
                            metadata={
                                'trade_type': 'compression',
                                'pair_id': pair_id,
                                'spread_level': current_spread,
                                'spread_velocity': spread_velocity,
                                'vol_zscore': vol_zscore,
                                'leg': 'far',
                                'near_market_id': near['market_id'],
                                'far_market_id': far['market_id']
                            }
                        ))

                        signals.append(Signal(
                            market_id=near['market_id'],
                            token_id=near['token_id'],
                            outcome='No',
                            signal_type=SignalType.BUY,
                            size=position_size,
                            price=near['price'],
                            reason=f"Spread compression: vol_spike (z={vol_zscore:.2f}) + spread_widening (Δ={spread_change:.4f})",
                            metadata={
                                'trade_type': 'compression',
                                'pair_id': pair_id,
                                'spread_level': current_spread,
                                'spread_velocity': spread_velocity,
                                'vol_zscore': vol_zscore,
                                'leg': 'near',
                                'near_market_id': near['market_id'],
                                'far_market_id': far['market_id']
                            }
                        ))

                        # Track position
                        self.spread_positions[position_key] = {
                            'trade_type': 'compression',
                            'entry_spread': current_spread,
                            'entry_date': current_date,
                            'entry_vol_zscore': vol_zscore,
                            'near_leg_size': position_size,
                            'far_leg_size': position_size
                        }

                # Widening trade: spread compressing + volume drought OR spread widening + inactive
                elif ((spread_velocity < 0 and volume_regime == 'inactive') or
                      (spread_velocity > 0 and volume_regime == 'inactive')):

                    position_size = self._calculate_position_size(
                        pair_id, abs(spread_velocity)
                    )

                    if position_size > 0:
                        # LONG far leg, SHORT near leg
                        signals.append(Signal(
                            market_id=far['market_id'],
                            token_id=far['token_id'],
                            outcome='No',
                            signal_type=SignalType.BUY,
                            size=position_size,
                            price=far['price'],
                            reason=f"Spread widening: vol_drought (z={vol_zscore:.2f}) + spread_velocity={spread_velocity:.4f}",
                            metadata={
                                'trade_type': 'widening',
                                'pair_id': pair_id,
                                'spread_level': current_spread,
                                'spread_velocity': spread_velocity,
                                'vol_zscore': vol_zscore,
                                'leg': 'far',
                                'near_market_id': near['market_id'],
                                'far_market_id': far['market_id']
                            }
                        ))

                        signals.append(Signal(
                            market_id=near['market_id'],
                            token_id=near['token_id'],
                            outcome='No',
                            signal_type=SignalType.SELL,
                            size=position_size,
                            price=near['price'],
                            reason=f"Spread widening: vol_drought (z={vol_zscore:.2f}) + spread_velocity={spread_velocity:.4f}",
                            metadata={
                                'trade_type': 'widening',
                                'pair_id': pair_id,
                                'spread_level': current_spread,
                                'spread_velocity': spread_velocity,
                                'vol_zscore': vol_zscore,
                                'leg': 'near',
                                'near_market_id': near['market_id'],
                                'far_market_id': far['market_id']
                            }
                        ))

                        # Track position
                        self.spread_positions[position_key] = {
                            'trade_type': 'widening',
                            'entry_spread': current_spread,
                            'entry_date': current_date,
                            'entry_vol_zscore': vol_zscore,
                            'near_leg_size': position_size,
                            'far_leg_size': position_size
                        }

        return signals

    def _calculate_position_size(self, pair_id: str, edge: float) -> float:
        """
        Calculate position size using Kelly criterion.

        Args:
            pair_id: Pair identifier
            edge: Expected spread change magnitude

        Returns:
            Position size as fraction of capital
        """
        # Get historical spread velocity variance if available
        if pair_id in self.spread_velocity_history and len(self.spread_velocity_history[pair_id]) > 5:
            velocities = self.spread_velocity_history[pair_id]
            edge_variance = np.var(velocities)
        else:
            # Conservative fallback: use edge/2
            edge_variance = (edge ** 2) / 4

        if edge_variance == 0:
            return self.max_position

        # Kelly fraction
        position_size = self.kelly_fraction * (edge / np.sqrt(edge_variance))
        position_size = min(position_size, self.max_position)
        position_size = max(position_size, 0.0)

        return position_size

    def _generate_close_signals(self, position_key: Tuple, near: Dict, far: Dict,
                                current_spread: float, reason: str) -> List[Signal]:
        """
        Generate signals to close both legs of a spread trade.

        Args:
            position_key: Position identifier
            near: Near contract data
            far: Far contract data
            current_spread: Current spread level
            reason: Exit reason

        Returns:
            List of close signals for both legs
        """
        signals = []
        position = self.spread_positions[position_key]

        # Close near leg (reverse the original trade)
        if position['trade_type'] == 'compression':
            # Original: LONG near -> Close with SELL
            signals.append(Signal(
                market_id=near['market_id'],
                token_id=near['token_id'],
                outcome='No',
                signal_type=SignalType.SELL,
                size=position['near_leg_size'],
                price=near['price'],
                reason=f"Close spread {position['trade_type']}: {reason}",
                metadata={
                    'exit_spread': current_spread,
                    'entry_spread': position['entry_spread'],
                    'leg': 'near'
                }
            ))
        else:  # widening
            # Original: SHORT near -> Close with BUY
            signals.append(Signal(
                market_id=near['market_id'],
                token_id=near['token_id'],
                outcome='No',
                signal_type=SignalType.BUY,
                size=position['near_leg_size'],
                price=near['price'],
                reason=f"Close spread {position['trade_type']}: {reason}",
                metadata={
                    'exit_spread': current_spread,
                    'entry_spread': position['entry_spread'],
                    'leg': 'near'
                }
            ))

        # Close far leg (reverse the original trade)
        if position['trade_type'] == 'compression':
            # Original: SHORT far -> Close with BUY
            signals.append(Signal(
                market_id=far['market_id'],
                token_id=far['token_id'],
                outcome='No',
                signal_type=SignalType.BUY,
                size=position['far_leg_size'],
                price=far['price'],
                reason=f"Close spread {position['trade_type']}: {reason}",
                metadata={
                    'exit_spread': current_spread,
                    'entry_spread': position['entry_spread'],
                    'leg': 'far'
                }
            ))
        else:  # widening
            # Original: LONG far -> Close with SELL
            signals.append(Signal(
                market_id=far['market_id'],
                token_id=far['token_id'],
                outcome='No',
                signal_type=SignalType.SELL,
                size=position['far_leg_size'],
                price=far['price'],
                reason=f"Close spread {position['trade_type']}: {reason}",
                metadata={
                    'exit_spread': current_spread,
                    'entry_spread': position['entry_spread'],
                    'leg': 'far'
                }
            ))

        # Remove position from tracking
        del self.spread_positions[position_key]

        return signals

    def on_data(self, current_date: pd.Timestamp, data: pd.DataFrame) -> List[Signal]:
        """
        Process new data and generate trading signals.

        Args:
            current_date: Current timestamp in backtest
            data: Market data up to current_date

        Returns:
            List of trading signals
        """
        signals = []

        # Get unique event groups
        if 'group_col' not in data.columns:
            return signals

        # OPTIMIZATION: Filter to recent window only (lookback + vol lookback)
        # We only need last N days for spread velocity and volume regime
        max_lookback = max(self.lookback_days, self.vol_lookback_days) + 2  # +2 buffer
        cutoff_date = current_date - pd.Timedelta(days=max_lookback)
        recent_data = data[data['date'] >= cutoff_date]

        # OPTIMIZATION: Pre-group by event to avoid repeated filtering
        # This reduces 6,572 full scans to 1 groupby operation
        grouped = recent_data.groupby('group_col')

        # Process each event group
        for event_id, event_data in grouped:
            if pd.isna(event_id):
                continue

            # Generate signals for this event (pass pre-filtered data)
            event_signals = self.generate_spread_signals(event_id, current_date, event_data)
            signals.extend(event_signals)

        return signals

    def on_market_close(self, market_id: str, outcome: str, final_price: float):
        """
        Called when one leg of a spread pair resolves.
        Immediately close the other leg.

        Args:
            market_id: Market that closed
            outcome: Winning outcome
            final_price: Final settlement price
        """
        super().on_market_close(market_id, outcome, final_price)

        # Find any spread positions containing this market
        positions_to_remove = []

        for position_key in list(self.spread_positions.keys()):
            event_id, near_market_id, far_market_id = position_key

            if market_id in [near_market_id, far_market_id]:
                # One leg has resolved - remove from tracking
                # The engine will handle the settlement
                positions_to_remove.append(position_key)

        for position_key in positions_to_remove:
            del self.spread_positions[position_key]

    def reset(self):
        """Reset strategy state."""
        super().reset()
        self.spread_positions = {}
        self.spread_history = {}
        self.volume_history = {}
        self.last_refit_date = None
        self.spread_velocity_history = {}
