"""
Kelly Criterion Position Sizing for Prediction Markets

Implements a custom Kelly criterion that adjusts position size based on:
1. Expected edge over market price
2. Priced carry (time decay / implied rate)
3. Model uncertainty (confidence interval width)

For a binary prediction market:
- Standard Kelly: f* = (p*b - q) / b = (p*(1/price - 1) - (1-p)) / (1/price - 1)
- Simplified: f* = edge / (1 - price) for binary markets
  where edge = model_prob - market_price

Adjustments:
- Carry adjustment: reduces position for markets with unfavorable carry
- Uncertainty discount: reduces position when model confidence is low
- Kelly fraction: typically use fractional Kelly (e.g., 0.25 = quarter Kelly)
"""

import numpy as np
from typing import Optional, Tuple


class KellyCriterion:
    """
    Kelly criterion position sizing for prediction markets.

    Attributes:
        kelly_fraction: Fraction of full Kelly to use (default: 0.25 for quarter Kelly)
        min_edge: Minimum edge required to open position (default: 0.05 = 5%)
        max_position: Maximum position size as fraction of capital (default: 0.15)
        carry_penalty: Penalty factor for unfavorable carry (default: 0.5)
    """

    def __init__(
        self,
        kelly_fraction: float = 0.25,
        min_edge: float = 0.05,
        max_position: float = 0.15,
        carry_penalty: float = 0.5
    ):
        """
        Initialize Kelly criterion calculator.

        Args:
            kelly_fraction: Fraction of full Kelly to use (0 < f <= 1)
            min_edge: Minimum edge to consider a position
            max_position: Maximum position size (safety cap)
            carry_penalty: How much to penalize unfavorable carry (0 to 1)
        """
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_position = max_position
        self.carry_penalty = carry_penalty

    def calculate_edge(
        self,
        model_prob: float,
        market_price: float,
        confidence_width: Optional[float] = None
    ) -> float:
        """
        Calculate expected edge with uncertainty adjustment.

        Args:
            model_prob: Model's predicted probability
            market_price: Current market price
            confidence_width: Width of confidence interval (optional)

        Returns:
            Adjusted edge accounting for model uncertainty
        """
        # Raw edge
        raw_edge = model_prob - market_price

        # Adjust for model uncertainty (wider CI = less confident = smaller edge)
        if confidence_width is not None and confidence_width > 0:
            # Penalize edge proportionally to uncertainty
            # If CI is 30% wide, reduce edge by ~30%
            uncertainty_factor = 1.0 - min(confidence_width, 0.5)
            adjusted_edge = raw_edge * uncertainty_factor
        else:
            adjusted_edge = raw_edge

        return adjusted_edge

    def calculate_carry_adjustment(
        self,
        market_price: float,
        time_to_expiration: float,
        risk_free_rate: float = 0.05
    ) -> float:
        """
        Calculate carry adjustment factor based on implied rate vs risk-free rate.

        Carry = implied_rate - risk_free_rate

        Positive carry (market yields more than risk-free) → boost position
        Negative carry (market yields less than risk-free) → reduce position

        Args:
            market_price: Current market price
            time_to_expiration: Time to expiration in years
            risk_free_rate: Risk-free rate (default: 5% = 0.05)

        Returns:
            Carry adjustment factor (0.5 to 1.5)
        """
        if time_to_expiration <= 0:
            return 1.0

        # Calculate implied rate: -ln(price) / time
        # For a "No" outcome, convert to Yes equivalent
        yes_price = 1.0 - market_price if market_price > 0.5 else market_price
        yes_price = np.clip(yes_price, 1e-6, 1 - 1e-6)

        implied_rate = -np.log(yes_price) / time_to_expiration

        # Calculate carry (excess return over risk-free)
        carry = implied_rate - risk_free_rate

        # Convert carry to adjustment factor
        # Positive carry → increase position (up to 1.5x)
        # Negative carry → decrease position (down to 0.5x)
        if carry > 0:
            # Boost position for positive carry (capped at +50%)
            carry_factor = 1.0 + min(carry * self.carry_penalty, 0.5)
        else:
            # Penalize negative carry (down to -50%)
            carry_factor = 1.0 + max(carry * self.carry_penalty, -0.5)

        return np.clip(carry_factor, 0.5, 1.5)

    def calculate_kelly_size(
        self,
        model_prob: float,
        market_price: float,
        confidence_width: Optional[float] = None,
        time_to_expiration: Optional[float] = None,
        portfolio_value: float = 10000.0,
        risk_free_rate: float = 0.05
    ) -> Tuple[float, dict]:
        """
        Calculate Kelly-optimal position size.

        Args:
            model_prob: Model's predicted probability (0 to 1)
            market_price: Current market price (0 to 1)
            confidence_width: CI width (optional, for uncertainty adjustment)
            time_to_expiration: Time to expiration in years (optional, for carry)
            portfolio_value: Current portfolio value in dollars
            risk_free_rate: Risk-free rate (default: 5%)

        Returns:
            Tuple of (position_fraction, metadata_dict)
            - position_fraction: Fraction of portfolio to allocate (0 to max_position)
            - metadata: Dict with edge, kelly_frac, carry_factor, etc.
        """
        # Calculate adjusted edge
        edge = self.calculate_edge(model_prob, market_price, confidence_width)

        # Check minimum edge threshold
        if abs(edge) < self.min_edge:
            return 0.0, {
                'edge': edge,
                'below_threshold': True,
                'kelly_fraction': 0.0,
                'carry_factor': 1.0
            }

        # Calculate carry adjustment
        carry_factor = 1.0
        if time_to_expiration is not None:
            carry_factor = self.calculate_carry_adjustment(
                market_price, time_to_expiration, risk_free_rate
            )

        # Calculate Kelly fraction
        # For binary market: f* = edge / (1 - price)
        # This is the fraction of capital to risk
        if edge > 0:
            # Long position: buying at market_price
            denominator = max(1.0 - market_price, 0.01)
            kelly_frac = edge / denominator
        else:
            # Short position: selling at market_price
            denominator = max(market_price, 0.01)
            kelly_frac = -edge / denominator

        # Apply fractional Kelly (e.g., quarter Kelly = 0.25)
        kelly_frac *= self.kelly_fraction

        # Apply carry adjustment
        kelly_frac *= carry_factor

        # Cap at maximum position size
        kelly_frac = min(abs(kelly_frac), self.max_position)

        # Preserve sign (positive = long, negative = short)
        kelly_frac = kelly_frac if edge > 0 else -kelly_frac

        metadata = {
            'edge': edge,
            'raw_edge': model_prob - market_price,
            'kelly_fraction': kelly_frac,
            'carry_factor': carry_factor,
            'confidence_width': confidence_width,
            'time_to_expiration': time_to_expiration,
            'position_dollars': abs(kelly_frac) * portfolio_value
        }

        return kelly_frac, metadata

    def calculate_multiple_positions(
        self,
        positions: list,
        portfolio_value: float,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> dict:
        """
        Calculate Kelly sizes for multiple correlated positions.

        For correlated positions, Kelly sizing needs adjustment.
        This is a simplified approach - full implementation would use
        covariance matrix of returns.

        Args:
            positions: List of dicts with 'model_prob', 'market_price', etc.
            portfolio_value: Current portfolio value
            correlation_matrix: Optional correlation matrix between positions

        Returns:
            Dict mapping position index to (size, metadata)
        """
        n_positions = len(positions)

        # Calculate individual Kelly fractions
        individual_sizes = []
        metadata_list = []

        for pos in positions:
            size, metadata = self.calculate_kelly_size(
                model_prob=pos.get('model_prob'),
                market_price=pos.get('market_price'),
                confidence_width=pos.get('confidence_width'),
                time_to_expiration=pos.get('time_to_expiration'),
                portfolio_value=portfolio_value,
                risk_free_rate=pos.get('risk_free_rate', 0.05)
            )
            individual_sizes.append(size)
            metadata_list.append(metadata)

        # If no correlation matrix, assume independent positions
        if correlation_matrix is None:
            # Independent positions: allocate full Kelly to each
            adjusted_sizes = individual_sizes
        else:
            # Correlated positions: reduce sizes
            # Simple heuristic: if avg correlation > 0.3, reduce by correlation factor
            avg_correlation = np.mean(np.abs(correlation_matrix[np.triu_indices(n_positions, k=1)]))
            correlation_discount = 1.0 - min(avg_correlation, 0.5)
            adjusted_sizes = [size * correlation_discount for size in individual_sizes]

        # Normalize if total allocation exceeds maximum
        total_allocation = sum(abs(s) for s in adjusted_sizes)
        if total_allocation > self.max_position:
            scale_factor = self.max_position / total_allocation
            adjusted_sizes = [size * scale_factor for size in adjusted_sizes]

        # Build result dict
        results = {}
        for i, (size, metadata) in enumerate(zip(adjusted_sizes, metadata_list)):
            metadata['adjusted_size'] = size
            metadata['correlation_discount'] = correlation_discount if correlation_matrix is not None else 1.0
            results[i] = (size, metadata)

        return results


def simple_kelly_size(
    model_prob: float,
    market_price: float,
    portfolio_value: float = 10000.0,
    kelly_fraction: float = 0.25,
    max_position: float = 0.15
) -> float:
    """
    Simple Kelly sizing (no carry adjustment).

    Convenience function for quick Kelly calculation.

    Args:
        model_prob: Model's predicted probability
        market_price: Current market price
        portfolio_value: Portfolio value in dollars
        kelly_fraction: Fractional Kelly (default: 0.25)
        max_position: Max position as fraction of portfolio

    Returns:
        Position size in dollars
    """
    edge = model_prob - market_price

    if abs(edge) < 0.05:  # Minimum 5% edge
        return 0.0

    if edge > 0:
        denominator = max(1.0 - market_price, 0.01)
        kelly_frac = edge / denominator
    else:
        denominator = max(market_price, 0.01)
        kelly_frac = -edge / denominator

    kelly_frac *= kelly_fraction
    kelly_frac = min(abs(kelly_frac), max_position)

    return abs(kelly_frac) * portfolio_value
