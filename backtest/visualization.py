"""
Visualization utilities for backtest results.
"""

import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional


def plot_backtest_results(results: Dict, output_file: Optional[str] = None) -> go.Figure:
    """
    Create comprehensive visualization of backtest results.

    Args:
        results: Backtest results dictionary
        output_file: Optional file path to save HTML plot

    Returns:
        Plotly figure
    """
    history = results['history']
    trades = results['trades']

    # Create subplots
    fig = make_subplots(
        rows=4, cols=2,
        subplot_titles=(
            'Portfolio Value Over Time',
            'Drawdown',
            'Daily Returns Distribution',
            'Cumulative Returns',
            'Number of Open Positions',
            'Trade P&L Distribution',
            'Win/Loss Ratio Over Time',
            'Monthly Returns Heatmap'
        ),
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": False}, {"type": "heatmap"}]
        ],
        vertical_spacing=0.12,
        horizontal_spacing=0.12
    )

    # 1. Portfolio value over time
    fig.add_trace(
        go.Scatter(
            x=history['date'],
            y=history['total_value'],
            name='Portfolio Value',
            line=dict(color='blue', width=2),
            hovertemplate='%{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ),
        row=1, col=1
    )

    # Add initial capital line
    fig.add_hline(
        y=results['initial_capital'],
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial Capital",
        row=1, col=1
    )

    # 2. Drawdown
    portfolio_values = history['total_value'].values
    cummax = np.maximum.accumulate(portfolio_values)
    drawdown = (portfolio_values - cummax) / cummax * 100

    fig.add_trace(
        go.Scatter(
            x=history['date'],
            y=drawdown,
            name='Drawdown',
            fill='tozeroy',
            line=dict(color='red', width=1),
            hovertemplate='%{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
        ),
        row=1, col=2
    )

    # 3. Daily returns distribution
    if 'daily_returns' in results and len(results['daily_returns']) > 0:
        returns = results['daily_returns'] * 100  # Convert to percentage

        fig.add_trace(
            go.Histogram(
                x=returns,
                name='Returns',
                nbinsx=50,
                marker_color='lightblue',
                hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
            ),
            row=2, col=1
        )

    # 4. Cumulative returns
    history['cumulative_return'] = (history['total_value'] / results['initial_capital'] - 1) * 100

    fig.add_trace(
        go.Scatter(
            x=history['date'],
            y=history['cumulative_return'],
            name='Cumulative Return',
            line=dict(color='green', width=2),
            hovertemplate='%{x}<br>Return: %{y:.2f}%<extra></extra>'
        ),
        row=2, col=2
    )

    # 5. Number of open positions
    fig.add_trace(
        go.Scatter(
            x=history['date'],
            y=history['num_positions'],
            name='Open Positions',
            line=dict(color='purple', width=2),
            mode='lines',
            hovertemplate='%{x}<br>Positions: %{y}<extra></extra>'
        ),
        row=3, col=1
    )

    # 6. Trade P&L distribution
    if not trades.empty:
        fig.add_trace(
            go.Histogram(
                x=trades['pnl'],
                name='Trade P&L',
                nbinsx=30,
                marker_color='lightgreen',
                hovertemplate='P&L: $%{x:.2f}<br>Count: %{y}<extra></extra>'
            ),
            row=3, col=2
        )

        # 7. Win/Loss ratio over time
        trades_sorted = trades.sort_values('exit_date')
        trades_sorted['cumulative_wins'] = (trades_sorted['pnl'] > 0).cumsum()
        trades_sorted['cumulative_losses'] = (trades_sorted['pnl'] < 0).cumsum()
        trades_sorted['win_loss_ratio'] = trades_sorted['cumulative_wins'] / (trades_sorted['cumulative_losses'] + 1)

        fig.add_trace(
            go.Scatter(
                x=trades_sorted['exit_date'],
                y=trades_sorted['win_loss_ratio'],
                name='Win/Loss Ratio',
                line=dict(color='orange', width=2),
                hovertemplate='%{x}<br>W/L Ratio: %{y:.2f}<extra></extra>'
            ),
            row=4, col=1
        )

        # 8. Monthly returns heatmap
        if not history.empty and 'date' in history.columns:
            history['year'] = history['date'].dt.year
            history['month'] = history['date'].dt.month

            # Calculate monthly returns
            monthly_returns = []
            for year in history['year'].unique():
                year_data = history[history['year'] == year]
                for month in range(1, 13):
                    month_data = year_data[year_data['month'] == month]
                    if not month_data.empty:
                        month_return = (
                            (month_data.iloc[-1]['total_value'] - month_data.iloc[0]['total_value']) /
                            month_data.iloc[0]['total_value'] * 100
                        )
                        monthly_returns.append({
                            'year': year,
                            'month': month,
                            'return': month_return
                        })

            if monthly_returns:
                monthly_df = pd.DataFrame(monthly_returns)
                pivot = monthly_df.pivot(index='year', columns='month', values='return')

                month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                             'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

                fig.add_trace(
                    go.Heatmap(
                        z=pivot.values,
                        x=month_names,
                        y=pivot.index,
                        colorscale='RdYlGn',
                        zmid=0,
                        hovertemplate='%{y} %{x}<br>Return: %{z:.2f}%<extra></extra>',
                        colorbar=dict(title="Return %")
                    ),
                    row=4, col=2
                )

    # Update layout
    fig.update_layout(
        height=1400,
        showlegend=False,
        title_text=f"Backtest Results: {results['strategy_name']}",
        title_font_size=20
    )

    # Update axes labels
    fig.update_xaxes(title_text="Date", row=1, col=1)
    fig.update_yaxes(title_text="Portfolio Value ($)", row=1, col=1)

    fig.update_xaxes(title_text="Date", row=1, col=2)
    fig.update_yaxes(title_text="Drawdown (%)", row=1, col=2)

    fig.update_xaxes(title_text="Daily Return (%)", row=2, col=1)
    fig.update_yaxes(title_text="Frequency", row=2, col=1)

    fig.update_xaxes(title_text="Date", row=2, col=2)
    fig.update_yaxes(title_text="Cumulative Return (%)", row=2, col=2)

    fig.update_xaxes(title_text="Date", row=3, col=1)
    fig.update_yaxes(title_text="Number of Positions", row=3, col=1)

    fig.update_xaxes(title_text="Trade P&L ($)", row=3, col=2)
    fig.update_yaxes(title_text="Frequency", row=3, col=2)

    fig.update_xaxes(title_text="Date", row=4, col=1)
    fig.update_yaxes(title_text="Win/Loss Ratio", row=4, col=1)

    fig.update_xaxes(title_text="Month", row=4, col=2)
    fig.update_yaxes(title_text="Year", row=4, col=2)

    # Save to file if specified
    if output_file:
        fig.write_html(output_file)

    return fig


def plot_equity_curve(history: pd.DataFrame, initial_capital: float) -> go.Figure:
    """
    Plot simple equity curve.

    Args:
        history: Portfolio history DataFrame
        initial_capital: Initial capital

    Returns:
        Plotly figure
    """
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history['date'],
        y=history['total_value'],
        mode='lines',
        name='Portfolio Value',
        line=dict(color='blue', width=2)
    ))

    fig.add_hline(
        y=initial_capital,
        line_dash="dash",
        line_color="gray",
        annotation_text="Initial Capital"
    )

    fig.update_layout(
        title="Portfolio Equity Curve",
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        hovermode='x unified'
    )

    return fig


def plot_trades_timeline(trades: pd.DataFrame) -> go.Figure:
    """
    Plot trades on a timeline.

    Args:
        trades: Trades DataFrame

    Returns:
        Plotly figure
    """
    if trades.empty:
        return go.Figure()

    # Color by P&L
    colors = ['green' if pnl > 0 else 'red' for pnl in trades['pnl']]

    fig = go.Figure()

    # Plot each trade as a line from entry to exit
    for _, trade in trades.iterrows():
        color = 'green' if trade['pnl'] > 0 else 'red'

        fig.add_trace(go.Scatter(
            x=[trade['entry_date'], trade['exit_date']],
            y=[trade['entry_price'], trade['exit_price']],
            mode='lines+markers',
            line=dict(color=color, width=2),
            marker=dict(size=8),
            name=f"Trade {trade['trade_id']}",
            hovertemplate=f"Market: {trade['market_id']}<br>" +
                         f"Entry: {trade['entry_date']}<br>" +
                         f"Exit: {trade['exit_date']}<br>" +
                         f"P&L: ${trade['pnl']:.2f}<extra></extra>"
        ))

    fig.update_layout(
        title="Trades Timeline",
        xaxis_title="Date",
        yaxis_title="Price",
        showlegend=False,
        hovermode='closest'
    )

    return fig
