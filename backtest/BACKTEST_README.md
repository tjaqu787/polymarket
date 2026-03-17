

## Creating Custom Strategies

To create a custom strategy, inherit from the `Strategy` base class:

```python
from backtest.strategy import Strategy, Signal, SignalType
import pandas as pd
from typing import List

class MyStrategy(Strategy):
    """My custom strategy."""

    def __init__(self, config=None):
        super().__init__(config)
        # Initialize strategy parameters
        self.my_param = self.config.get('my_param', 1.0)

    @property
    def name(self) -> str:
        return "MyStrategy"

    def on_data(self, current_date: pd.Timestamp, data: pd.DataFrame) -> List[Signal]:
        """Generate trading signals based on current data."""
        signals = []

        # Get data for current date
        today_data = data[data['date'] == current_date]

        # Implement your logic here
        for _, row in today_data.iterrows():
            if row['outcome'] == 'Yes':
                # Example: buy if price is low
                if row['price'] < 0.5:
                    signals.append(Signal(
                        market_id=row['market_id'],
                        token_id=row['token_id'],
                        outcome='Yes',
                        signal_type=SignalType.BUY,
                        size=1.0,
                        price=row['price'],
                        reason="Price below 0.5"
                    ))

        return signals
```

## Performance Metrics

The backtesting engine calculates comprehensive performance metrics:

### Returns
- **Total Return**: Overall percentage gain/loss
- **CAGR**: Compound Annual Growth Rate
- **Daily Returns**: Day-over-day returns series

### Risk Metrics
- **Sharpe Ratio**: Risk-adjusted return (higher is better)
- **Sortino Ratio**: Like Sharpe but only penalizes downside volatility
- **Volatility**: Standard deviation of returns (annualized)
- **Max Drawdown**: Largest peak-to-trough decline
- **Max Drawdown Duration**: Longest underwater period in days
- **Calmar Ratio**: CAGR / Max Drawdown

### Trade Statistics
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Average Win**: Mean P&L of winning trades
- **Average Loss**: Mean P&L of losing trades
- **Average Trade**: Mean P&L across all trades
- **Expectancy**: Expected value per trade

## Visualization

The visualization module creates interactive plots showing:

1. **Portfolio Value Over Time**: Equity curve with initial capital reference
2. **Drawdown**: Underwater equity chart
3. **Daily Returns Distribution**: Histogram of daily returns
4. **Cumulative Returns**: Running total return percentage
5. **Number of Open Positions**: Position count over time
6. **Trade P&L Distribution**: Histogram of trade profits/losses
7. **Win/Loss Ratio Over Time**: Evolution of win/loss ratio
8. **Monthly Returns Heatmap**: Calendar view of monthly performance

All plots are interactive (zoom, pan, hover) and saved as HTML files.

## Troubleshooting

### No data loaded
- Check that `data/polymarket.db` exists and has data
- Verify date range overlaps with available data
- Check volume filters aren't too restrictive

### No trades executed
- Strategy may be too conservative
- Check that signals are being generated (add logging)
- Verify position size calculations
- Check if max_positions limit is being hit

### Poor performance
- Try different parameter values
- Check if strategy logic matches market behavior
- Compare against buy-and-hold baseline
- Consider transaction costs

## Future Enhancements

Potential additions:
- More sophisticated position sizing (Kelly criterion, risk parity)
- Stop-loss and take-profit orders
- Market-making strategies
- Pair trading across related markets
- Machine learning-based strategies
- Live trading integration
- Multi-asset portfolio optimization

## License

Part of the Polymarket analysis project for Bayesian Statistics course.
