"""
Streamlit visualization dashboard for backtest results.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from typing import Dict, Optional


def create_backtest_dashboard(results: Dict) -> None:
    """
    Create Streamlit dashboard for backtest results.

    Args:
        results: Backtest results dictionary
    """
    st.title(f"Backtest Results: {results['strategy_name']}")

    history = results['history']
    trades = results['trades']

    # Key metrics in columns
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Final Portfolio Value",
            f"${results['final_value']:,.2f}",
            f"{results['total_return']:.2f}%"
        )

    with col2:
        st.metric(
            "Total Return",
            f"{results['total_return']:.2f}%",
            f"{results['annualized_return']:.2f}% annualized"
        )

    with col3:
        st.metric(
            "Sharpe Ratio",
            f"{results['sharpe_ratio']:.2f}"
        )

    with col4:
        st.metric(
            "Max Drawdown",
            f"{results['max_drawdown']:.2f}%"
        )

    # Additional metrics
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        st.metric("Total Trades", results['num_trades'])

    with col6:
        st.metric("Win Rate", f"{results['win_rate']:.1f}%")

    with col7:
        st.metric("Avg Win", f"${results['average_win']:.2f}")

    with col8:
        st.metric("Avg Loss", f"${results['average_loss']:.2f}")

    st.divider()

    # Portfolio Value and Drawdown
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Portfolio Value Over Time")
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=history['date'],
            y=history['total_value'],
            name='Portfolio Value',
            line=dict(color='blue', width=2),
            hovertemplate='%{x}<br>Value: $%{y:,.2f}<extra></extra>'
        ))
        fig1.add_hline(
            y=results['initial_capital'],
            line_dash="dash",
            line_color="gray",
            annotation_text="Initial Capital"
        )
        fig1.update_layout(xaxis_title="Date", yaxis_title="Portfolio Value ($)", height=400)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Drawdown")
        portfolio_values = history['total_value'].values
        cummax = np.maximum.accumulate(portfolio_values)
        drawdown = (portfolio_values - cummax) / cummax * 100

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=history['date'],
            y=drawdown,
            fill='tozeroy',
            line=dict(color='red', width=1),
            hovertemplate='%{x}<br>Drawdown: %{y:.2f}%<extra></extra>'
        ))
        fig2.update_layout(xaxis_title="Date", yaxis_title="Drawdown (%)", height=400)
        st.plotly_chart(fig2, use_container_width=True)

    # Returns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Daily Returns Distribution")
        if 'daily_returns' in results and len(results['daily_returns']) > 0:
            returns = results['daily_returns'] * 100
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(
                x=returns,
                nbinsx=50,
                marker_color='lightblue',
                hovertemplate='Return: %{x:.2f}%<br>Count: %{y}<extra></extra>'
            ))
            fig3.update_layout(xaxis_title="Daily Return (%)", yaxis_title="Frequency", height=400)
            st.plotly_chart(fig3, use_container_width=True)

    with col2:
        st.subheader("Cumulative Returns")
        history_copy = history.copy()
        history_copy['cumulative_return'] = (history_copy['total_value'] / results['initial_capital'] - 1) * 100

        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=history_copy['date'],
            y=history_copy['cumulative_return'],
            line=dict(color='green', width=2),
            hovertemplate='%{x}<br>Return: %{y:.2f}%<extra></extra>'
        ))
        fig4.update_layout(xaxis_title="Date", yaxis_title="Cumulative Return (%)", height=400)
        st.plotly_chart(fig4, use_container_width=True)

    # Positions and Trades
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Number of Open Positions")
        fig5 = go.Figure()
        fig5.add_trace(go.Scatter(
            x=history['date'],
            y=history['num_positions'],
            line=dict(color='purple', width=2),
            hovertemplate='%{x}<br>Positions: %{y}<extra></extra>'
        ))
        fig5.update_layout(xaxis_title="Date", yaxis_title="Number of Positions", height=400)
        st.plotly_chart(fig5, use_container_width=True)

    with col2:
        st.subheader("Trade P&L Distribution")
        if not trades.empty:
            fig6 = go.Figure()
            fig6.add_trace(go.Histogram(
                x=trades['pnl'],
                nbinsx=30,
                marker_color='lightgreen',
                hovertemplate='P&L: $%{x:.2f}<br>Count: %{y}<extra></extra>'
            ))
            fig6.update_layout(xaxis_title="Trade P&L ($)", yaxis_title="Frequency", height=400)
            st.plotly_chart(fig6, use_container_width=True)

    # Win/Loss Ratio and Monthly Returns
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Win/Loss Ratio Over Time")
        if not trades.empty:
            trades_sorted = trades.sort_values('exit_date')
            trades_sorted['cumulative_wins'] = (trades_sorted['pnl'] > 0).cumsum()
            trades_sorted['cumulative_losses'] = (trades_sorted['pnl'] < 0).cumsum()
            trades_sorted['win_loss_ratio'] = trades_sorted['cumulative_wins'] / (trades_sorted['cumulative_losses'] + 1)

            fig7 = go.Figure()
            fig7.add_trace(go.Scatter(
                x=trades_sorted['exit_date'],
                y=trades_sorted['win_loss_ratio'],
                line=dict(color='orange', width=2),
                hovertemplate='%{x}<br>W/L Ratio: %{y:.2f}<extra></extra>'
            ))
            fig7.update_layout(xaxis_title="Date", yaxis_title="Win/Loss Ratio", height=400)
            st.plotly_chart(fig7, use_container_width=True)

    with col2:
        st.subheader("Monthly Returns Heatmap")
        if not history.empty and 'date' in history.columns:
            history_copy = history.copy()
            history_copy['year'] = history_copy['date'].dt.year
            history_copy['month'] = history_copy['date'].dt.month

            monthly_returns = []
            for year in history_copy['year'].unique():
                year_data = history_copy[history_copy['year'] == year]
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

                fig8 = go.Figure()
                fig8.add_trace(go.Heatmap(
                    z=pivot.values,
                    x=month_names,
                    y=pivot.index,
                    colorscale='RdYlGn',
                    zmid=0,
                    hovertemplate='%{y} %{x}<br>Return: %{z:.2f}%<extra></extra>',
                    colorbar=dict(title="Return %")
                ))
                fig8.update_layout(xaxis_title="Month", yaxis_title="Year", height=400)
                st.plotly_chart(fig8, use_container_width=True)

    # Trades table
    st.divider()
    st.subheader("Trade History")
    if not trades.empty:
        st.dataframe(
            trades[['trade_id', 'market_id', 'side', 'entry_date', 'exit_date',
                   'entry_price', 'exit_price', 'quantity', 'pnl']].sort_values('exit_date', ascending=False),
            use_container_width=True
        )


def plot_backtest_results(results: Dict, output_file: Optional[str] = None) -> go.Figure:
    """
    Create comprehensive visualization of backtest results (legacy function).

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


def main():
    """
    Main Streamlit app function.
    Run with: streamlit run backtest/visualization.py
    """
    st.set_page_config(
        page_title="Backtest Results Dashboard",
        page_icon="\U0001F4C8",
        layout="wide"
    )

    st.sidebar.title("Backtest Dashboard")
    st.sidebar.info("Upload backtest results or use sample data")

    # Option to upload results file
    uploaded_file = st.sidebar.file_uploader(
        "Upload backtest results (pickle or JSON)",
        type=['pkl', 'pickle', 'json']
    )

    if uploaded_file is not None:
        import pickle
        import json

        try:
            if uploaded_file.name.endswith('.json'):
                results = json.load(uploaded_file)
                # Convert date strings to datetime if needed
                if 'history' in results:
                    results['history'] = pd.DataFrame(results['history'])
                    if 'date' in results['history'].columns:
                        results['history']['date'] = pd.to_datetime(results['history']['date'])
                if 'trades' in results:
                    results['trades'] = pd.DataFrame(results['trades'])
                    if 'entry_date' in results['trades'].columns:
                        results['trades']['entry_date'] = pd.to_datetime(results['trades']['entry_date'])
                        results['trades']['exit_date'] = pd.to_datetime(results['trades']['exit_date'])
            else:
                results = pickle.load(uploaded_file)

            create_backtest_dashboard(results)

        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.exception(e)
    else:
        st.info("Please upload a backtest results file to view the dashboard")
        st.markdown("""
        ### How to use:
        1. Run a backtest using the backtest engine
        2. Save the results to a pickle or JSON file
        3. Upload the file using the sidebar
        4. View your interactive dashboard!

        ### Example:
        ```python
        from backtest.engine import BacktestEngine
        import pickle

        # Run backtest
        engine = BacktestEngine(...)
        results = engine.run()

        # Save results
        with open('results.pkl', 'wb') as f:
            pickle.dump(results, f)
        ```
        """)


if __name__ == "__main__":
    main()
