"""
Streamlit dashboard for visualizing implied continuous interest rates
from Polymarket prediction markets over time.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
sys.path.append('models')
sys.path.append('utils')
from implied_rates import (
    get_market_groups,
    load_price_history,
    calculate_implied_rates_for_market_group
)
from term_structure import (
    extract_term_structure_history,
    calculate_term_structure_metrics,
    TermStructure
)


st.set_page_config(
    page_title="Polymarket Implied Rates Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Polymarket Hazard Rate Dashboard")
st.markdown("""
This dashboard shows implied hazard rates from Polymarket prediction markets.
The hazard rate is calculated as: **λ = -ln(1 - P(T)) / T**

Where:
- P(T) = market probability that event occurs by time T
- T = time to expiration in years
- λ = implied hazard rate (intensity of event occurrence)
""")

# Load market groups
@st.cache_data
def load_groups():
    return get_market_groups()

# Add cache clear button
if st.sidebar.button("🔄 Clear Cache"):
    st.cache_data.clear()
    st.rerun()

groups_df = load_groups()

if len(groups_df) == 0:
    st.error("No market groups found with multiple resolution dates.")
    st.stop()

# Sidebar for market selection
st.sidebar.header("Market Selection")

# Create a readable display name for each market group
# Check if canonical_slug exists (new semantic grouping)
if 'canonical_slug' in groups_df.columns:
    display_base = groups_df['canonical_slug'].fillna(groups_df['event_slug']).fillna(groups_df['market_group'])
    actor_suffix = groups_df['actor'].apply(lambda x: f" ({x})" if pd.notna(x) else "")
else:
    # Fallback for old grouping
    display_base = groups_df['event_title'].fillna(groups_df['event_slug'])
    actor_suffix = ""

groups_df['display_name'] = (
    display_base + actor_suffix +
    " [" + groups_df['num_markets'].astype(str) + " markets, " +
    groups_df['num_dates'].astype(str) + " dates]"
)

selected_display = st.sidebar.selectbox(
    "Select a market group:",
    groups_df['display_name'].tolist(),
    index=0
)

# Get the selected market_group
selected_idx = groups_df[groups_df['display_name'] == selected_display].index[0]
selected_group = groups_df.loc[selected_idx, 'market_group']
selected_event_slug = groups_df.loc[selected_idx, 'event_slug']
selected_event_title = groups_df.loc[selected_idx, 'event_title']
# Display market info
st.sidebar.markdown("---")
st.sidebar.markdown("**Market Details:**")
st.sidebar.write(f"**Event:** {selected_event_title}")

# Show semantic grouping info if available
if 'canonical_slug' in groups_df.columns:
    selected_canonical_slug = groups_df.loc[selected_idx, 'canonical_slug']
    selected_actor = groups_df.loc[selected_idx, 'actor']
    st.sidebar.write(f"**Canonical Slug:** {selected_canonical_slug}")
    if pd.notna(selected_actor):
        st.sidebar.write(f"**Actor/Country:** {selected_actor.upper()}")
    st.sidebar.write(f"**Semantic Group ID:** {selected_group}")
else:
    st.sidebar.write(f"**Group ID:** {selected_group}")

st.sidebar.write(f"**Event Slug:** {selected_event_slug}")
st.sidebar.write(f"**Markets:** {groups_df.loc[selected_idx, 'num_markets']}")
st.sidebar.write(f"**Resolution dates:** {groups_df.loc[selected_idx, 'num_dates']}")

# Load price history for selected market
@st.cache_data
def load_data(market_group):
    df = load_price_history(market_group=market_group)
    if len(df) > 0:
        df = calculate_implied_rates_for_market_group(df)
    return df

with st.spinner(f"Loading data for {selected_event_title}..."):
    data = load_data(selected_group)

if len(data) == 0:
    st.warning(f"No price history data available for this market group. "
               f"You may need to run `data/downloaders/get_pricing.py` to fetch price data.")
    st.stop()

# Add some metrics
st.sidebar.markdown("---")
st.sidebar.markdown("**Data Summary:**")
st.sidebar.write(f"Total price points: {len(data):,}")
st.sidebar.write(f"Markets: {data['market_id'].nunique()}")
st.sidebar.write(f"Date range: {data['date'].min()} to {data['date'].max()}")

# Main content area
tab1, tab2, tab3, tab4 = st.tabs(["⏱️ Time-to-Event Estimate", "📈 Price Evolution", "📉 Term Structure", "📋 Raw Data"])

with tab1:
    st.header("Time-to-Event Estimate with Gamma Prior")

    # Outcome selector - check what outcomes are available in the data
    available_outcomes = sorted(data['outcome'].unique())

    if len(available_outcomes) == 0:
        st.error("No outcome data available")
        st.stop()

    # Default to the first available outcome
    outcome_filter = st.radio(
        "Select outcome to display:",
        available_outcomes,
        horizontal=True,
        index=0
    )

    # Show info about the data
    if len(available_outcomes) == 1:
        st.info(f"ℹ️ This view is filtered to '{available_outcomes[0]}' outcomes only (better for hazard rate modeling of unlikely events)")

    # Filter data based on outcome selection
    plot_data = data[data['outcome'] == outcome_filter].copy()

    # Get unique resolution dates
    resolution_dates = sorted(plot_data['resolution_date'].unique())

    # Color palette
    colors = px.colors.qualitative.Plotly

    # Create figure
    fig = go.Figure()

    for i, res_date in enumerate(resolution_dates):
        subset = plot_data[plot_data['resolution_date'] == res_date].sort_values('date')

        if len(subset) == 0:
            continue

        # Calculate time-to-event estimates for each observation
        # Using exponential distribution: E[T] = 1/λ where λ is hazard rate
        # For gamma prior on λ with shape α and rate β:
        # Posterior mean of T = β/(α-1) for α > 1
        # We'll use empirical Bayes: estimate α, β from the data

        hazard_rates = subset['implied_rate'].values
        dates = pd.to_datetime(subset['date'])
        resolution_dt = pd.to_datetime(res_date)

        # Filter out invalid rates
        valid_mask = np.isfinite(hazard_rates) & (hazard_rates > 0)
        hazard_rates = hazard_rates[valid_mask]
        dates = dates[valid_mask]

        if len(hazard_rates) == 0:
            continue

        # Estimate time-to-event using 1/λ (exponential mean)
        mean_time_to_event = 1.0 / hazard_rates  # in years

        # For confidence intervals, use a rolling window approach
        # This captures the variation in estimates over time rather than assuming a global distribution
        window_size = max(5, len(mean_time_to_event) // 10)  # adaptive window

        # Compute rolling quantiles for confidence intervals
        def rolling_quantile(arr, quantile, window):
            """Compute rolling quantile with padding"""
            result = np.full_like(arr, np.nan, dtype=float)
            for i in range(len(arr)):
                start = max(0, i - window // 2)
                end = min(len(arr), i + window // 2 + 1)
                window_data = arr[start:end]
                if len(window_data) > 0:
                    result[i] = np.percentile(window_data, quantile * 100)
            return result

        # Calculate rolling CIs
        lower_95 = rolling_quantile(mean_time_to_event, 0.025, window_size)
        upper_95 = rolling_quantile(mean_time_to_event, 0.975, window_size)
        lower_70 = rolling_quantile(mean_time_to_event, 0.15, window_size)
        upper_70 = rolling_quantile(mean_time_to_event, 0.85, window_size)

        # Cap all values at reasonable limits (max 5 years into future)
        mean_time_to_event = np.clip(mean_time_to_event, 0, 5.0)
        lower_95 = np.clip(lower_95, 0, 5.0)
        upper_95 = np.clip(upper_95, 0, 5.0)
        lower_70 = np.clip(lower_70, 0, 5.0)
        upper_70 = np.clip(upper_70, 0, 5.0)

        # Convert to actual dates (observation date + time-to-event)
        mean_event_dates = dates + pd.to_timedelta(mean_time_to_event * 365.25, unit='D')
        lower_70_dates = dates + pd.to_timedelta(lower_70 * 365.25, unit='D')
        upper_70_dates = dates + pd.to_timedelta(upper_70 * 365.25, unit='D')
        lower_95_dates = dates + pd.to_timedelta(lower_95 * 365.25, unit='D')
        upper_95_dates = dates + pd.to_timedelta(upper_95 * 365.25, unit='D')

        color = colors[i % len(colors)]

        # Plot 95% CI band
        fig.add_trace(go.Scatter(
            x=dates,
            y=upper_95_dates,
            mode='lines',
            name=f"{res_date} 95% CI",
            line=dict(width=0, color=color),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=dates,
            y=lower_95_dates,
            mode='lines',
            name=f"{res_date} 95% CI",
            line=dict(width=0, color=color),
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.2)',
            fill='tonexty',
            showlegend=True,
            hoverinfo='skip'
        ))

        # Plot 70% CI band
        fig.add_trace(go.Scatter(
            x=dates,
            y=upper_70_dates,
            mode='lines',
            name=f"{res_date} 70% CI",
            line=dict(width=0, color=color),
            showlegend=False,
            hoverinfo='skip'
        ))
        fig.add_trace(go.Scatter(
            x=dates,
            y=lower_70_dates,
            mode='lines',
            name=f"{res_date} 70% CI",
            line=dict(width=0, color=color),
            fillcolor=f'rgba({int(color[1:3], 16)}, {int(color[3:5], 16)}, {int(color[5:7], 16)}, 0.4)',
            fill='tonexty',
            showlegend=True,
            hoverinfo='skip'
        ))

        # Plot mean estimate
        fig.add_trace(go.Scatter(
            x=dates,
            y=mean_event_dates,
            mode='lines+markers',
            name=f"{res_date} Mean",
            line=dict(color=color, width=2),
            marker=dict(size=4, color=color),
            hovertemplate=(
                f"<b>{res_date}</b><br>" +
                "Observation Date: %{x}<br>" +
                "Estimated Event Date: %{y}<br>" +
                "<extra></extra>"
            )
        ))

        # Add horizontal line at actual resolution date
        fig.add_trace(go.Scatter(
            x=[dates.min(), dates.max()],
            y=[resolution_dt, resolution_dt],
            mode='lines',
            name=f"{res_date} Actual",
            line=dict(color=color, dash='dash', width=2),
            hovertemplate=f"<b>Actual Resolution: {res_date}</b><extra></extra>",
            showlegend=True
        ))

    fig.update_layout(
        title=f"Time-to-Event Estimates for {selected_event_title} ({outcome_filter})",
        xaxis_title="Observation Date",
        yaxis_title="Estimated Event Date",
        hovermode='closest',
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    st.plotly_chart(fig, use_container_width=True)

    # Add explanation
    st.markdown("""
    **How to interpret this chart:**
    - **Mean line**: Expected event date based on market hazard rate λ, using E[T] = 1/λ
    - **70% CI band** (darker): 70% confidence interval using rolling window of estimates
    - **95% CI band** (lighter): 95% confidence interval using rolling window of estimates
    - **Horizontal dashed line**: Actual resolution date (ground truth)
    - **Good performance**: Mean estimate converges to the actual date as time progresses
    - **Poor performance**: Mean estimate stays far from actual date or diverges over time

    The horizontal line lets you visually assess prediction quality by comparing:
    - How close the mean gets to the actual date
    - Whether the CI bands contain the actual date
    - How quickly the estimate converges
    """)

with tab2:
    st.header("Market Probability Evolution")

    # Outcome selector for price tab
    outcome_filter_price = st.radio(
        "Select outcome:",
        ["Yes", "No", "Both"],
        horizontal=True,
        key="price_outcome"
    )

    # Filter data
    if outcome_filter_price == "Both":
        plot_data_price = data.copy()
    else:
        plot_data_price = data[data['outcome'] == outcome_filter_price].copy()

    # Create price evolution plot
    fig_price = go.Figure()

    resolution_dates = sorted(plot_data_price['resolution_date'].unique())

    for i, res_date in enumerate(resolution_dates):
        for outcome in ['Yes', 'No']:
            if outcome_filter_price != "Both" and outcome != outcome_filter_price:
                continue

            subset = plot_data_price[
                (plot_data_price['resolution_date'] == res_date) &
                (plot_data_price['outcome'] == outcome)
            ].sort_values('date')

            if len(subset) == 0:
                continue

            line_style = 'solid' if outcome == 'Yes' else 'dash'
            color = colors[i % len(colors)]

            fig_price.add_trace(go.Scatter(
                x=subset['date'],
                y=subset['price'],
                mode='lines+markers',
                name=f"{res_date} ({outcome})",
                line=dict(dash=line_style, color=color),
                marker=dict(size=4, color=color),
                hovertemplate=(
                    f"<b>{res_date} ({outcome})</b><br>" +
                    "Date: %{x}<br>" +
                    "Probability: %{y:.2%}<br>" +
                    "<extra></extra>"
                )
            ))

    fig_price.update_layout(
        title=f"Market Probabilities for {selected_event_title}",
        xaxis_title="Date",
        yaxis_title="Probability",
        yaxis_tickformat=".0%",
        yaxis_range=[0, 1],
        hovermode='closest',
        height=600,
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02
        )
    )

    st.plotly_chart(fig_price, use_container_width=True)

with tab3:
    st.header("Term Structure Analysis")

    st.markdown("""
    **Term structure analysis** examines how implied rates vary across different time horizons.
    - **Level**: Average rate across all maturities
    - **Slope**: Difference between long-term and short-term rates (positive = upward sloping)
    - **Curvature**: Measure of the bend in the term structure
    """)

    # Outcome selector for term structure
    outcome_filter_ts = st.radio(
        "Select outcome for term structure:",
        ["Yes", "No"],
        horizontal=True,
        key="ts_outcome"
    )

    # Extract term structures for all dates
    with st.spinner("Calculating term structure metrics..."):
        term_structures = extract_term_structure_history(data, outcome_filter=outcome_filter_ts)

        if len(term_structures) == 0:
            st.warning("No term structure data available for the selected outcome.")
        else:
            metrics_df = calculate_term_structure_metrics(term_structures)

            # Plot 1: Term structure metrics over time
            st.subheader("Term Structure Metrics Over Time")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Latest Level",
                    f"{metrics_df['level'].iloc[-1]*100:.2f}%",
                    delta=f"{(metrics_df['level'].iloc[-1] - metrics_df['level'].iloc[0])*100:.2f}%" if len(metrics_df) > 1 else None
                )
            with col2:
                st.metric(
                    "Latest Slope",
                    f"{metrics_df['slope'].iloc[-1]*100:.2f}%",
                    delta=f"{(metrics_df['slope'].iloc[-1] - metrics_df['slope'].iloc[0])*100:.2f}%" if len(metrics_df) > 1 else None
                )
            with col3:
                st.metric(
                    "Latest Curvature",
                    f"{metrics_df['curvature'].iloc[-1]*100:.2f}%",
                    delta=f"{(metrics_df['curvature'].iloc[-1] - metrics_df['curvature'].iloc[0])*100:.2f}%" if len(metrics_df) > 1 else None
                )

            # Create metrics chart
            fig_metrics = go.Figure()

            fig_metrics.add_trace(go.Scatter(
                x=metrics_df['date'],
                y=metrics_df['level'],
                mode='lines+markers',
                name='Level',
                line=dict(color='blue'),
                yaxis='y'
            ))

            fig_metrics.add_trace(go.Scatter(
                x=metrics_df['date'],
                y=metrics_df['slope'],
                mode='lines+markers',
                name='Slope',
                line=dict(color='red'),
                yaxis='y'
            ))

            fig_metrics.add_trace(go.Scatter(
                x=metrics_df['date'],
                y=metrics_df['curvature'],
                mode='lines+markers',
                name='Curvature',
                line=dict(color='green'),
                yaxis='y'
            ))

            fig_metrics.update_layout(
                title=f"Term Structure Metrics - {outcome_filter_ts} Outcome",
                xaxis_title="Date",
                yaxis_title="Rate",
                yaxis_tickformat=".1%",
                hovermode='x unified',
                height=500,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(fig_metrics, use_container_width=True)

            # Plot 2: Term structure curves for selected dates
            st.subheader("Term Structure Curves")

            # Let user select dates to compare
            available_dates = sorted(term_structures.keys())

            # Default to showing up to 5 recent dates
            default_dates = available_dates[-5:] if len(available_dates) >= 5 else available_dates

            selected_dates = st.multiselect(
                "Select dates to display term structure curves:",
                available_dates,
                default=default_dates
            )

            if len(selected_dates) > 0:
                fig_curves = go.Figure()

                colors_ts = px.colors.qualitative.Set2

                for i, date in enumerate(selected_dates):
                    ts = term_structures[date]
                    color = colors_ts[i % len(colors_ts)]

                    # Sort by maturity for proper line plotting
                    sort_idx = np.argsort(ts.maturities)

                    fig_curves.add_trace(go.Scatter(
                        x=ts.maturities[sort_idx],
                        y=ts.rates[sort_idx],
                        mode='lines+markers',
                        name=date,
                        line=dict(color=color),
                        marker=dict(size=8, color=color),
                        hovertemplate=(
                            f"<b>{date}</b><br>" +
                            "Time to Maturity: %{x:.3f} years<br>" +
                            "Implied Rate: %{y:.2%}<br>" +
                            "<extra></extra>"
                        )
                    ))

                fig_curves.update_layout(
                    title=f"Term Structure Curves - {outcome_filter_ts} Outcome",
                    xaxis_title="Time to Maturity (Years)",
                    yaxis_title="Implied Rate",
                    yaxis_tickformat=".1%",
                    hovermode='closest',
                    height=500,
                    legend=dict(
                        orientation="v",
                        yanchor="top",
                        y=1,
                        xanchor="left",
                        x=1.02
                    )
                )

                st.plotly_chart(fig_curves, use_container_width=True)

                st.markdown("""
                **Interpretation:**
                - **Upward sloping** (positive slope): Short-term rates < long-term rates, typical for normal markets
                - **Flat**: Similar rates across all maturities
                - **Inverted** (negative slope): Short-term rates > long-term rates, may signal uncertainty
                - **Humped** (positive curvature): Medium-term rates higher than both short and long term
                """)

            # Show summary statistics
            with st.expander("View Term Structure Statistics"):
                st.dataframe(metrics_df, use_container_width=True)

with tab4:
    st.header("Raw Data")

    # Display raw data with filters
    st.markdown("**Filter options:**")
    col1, col2 = st.columns(2)

    with col1:
        selected_resolution = st.multiselect(
            "Resolution dates:",
            sorted(data['resolution_date'].unique()),
            default=sorted(data['resolution_date'].unique())[:3] if len(data['resolution_date'].unique()) > 0 else []
        )

    with col2:
        selected_outcome = st.multiselect(
            "Outcomes:",
            ['Yes', 'No'],
            default=['Yes', 'No']
        )

    # Filter data
    filtered_data = data[
        (data['resolution_date'].isin(selected_resolution)) &
        (data['outcome'].isin(selected_outcome))
    ].copy()

    # Select columns to display
    display_cols = [
        'date', 'resolution_date', 'outcome', 'price',
        'time_to_expiration', 'implied_rate', 'complement_implied_rate',
        'question', 'market_id'
    ]

    st.dataframe(
        filtered_data[display_cols].sort_values(['resolution_date', 'date', 'outcome']),
        use_container_width=True,
        height=400
    )

    # Download button
    csv = filtered_data[display_cols].to_csv(index=False)
    st.download_button(
        label="Download data as CSV",
        data=csv,
        file_name=f"implied_rates_{selected_event_slug}_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# Footer
st.markdown("---")
st.markdown("""
<small>
Data source: Polymarket | Dashboard created with Streamlit
</small>
""", unsafe_allow_html=True)
