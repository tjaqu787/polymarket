"""
Streamlit dashboard for visualizing implied continuous interest rates
from Polymarket prediction markets over time.
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
sys.path.append('models')
from implied_rates import (
    get_market_groups,
    load_price_history,
    calculate_implied_rates_for_market_group
)


st.set_page_config(
    page_title="Polymarket Implied Rates Dashboard",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Polymarket Implied Interest Rates Dashboard")
st.markdown("""
This dashboard shows implied continuous interest rates from Polymarket prediction markets.
The implied rate is calculated as: **r = -ln(p) / T**

Where:
- p = market probability (price)
- T = time to expiration in years
- r = implied annual continuous interest rate
""")

# Load market groups
@st.cache_data
def load_groups():
    return get_market_groups()

groups_df = load_groups()

if len(groups_df) == 0:
    st.error("No market groups found with multiple resolution dates.")
    st.stop()

# Sidebar for market selection
st.sidebar.header("Market Selection")

# Create a readable display name for each market group
groups_df['display_name'] = (
    groups_df['event_title'].fillna(groups_df['event_slug']) +
    " (" + groups_df['num_dates'].astype(str) + " dates)"
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
st.sidebar.write(f"Event: {selected_event_title}")
st.sidebar.write(f"Slug: {selected_event_slug}")
st.sidebar.write(f"Group ID: {selected_group}")
st.sidebar.write(f"Number of resolution dates: {groups_df.loc[selected_idx, 'num_dates']}")

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
tab1, tab2, tab3 = st.tabs(["📊 Implied Rates Over Time", "📈 Price Evolution", "📋 Raw Data"])

with tab1:
    st.header("Implied Interest Rates by Resolution Date")

    # Outcome selector
    outcome_filter = st.radio(
        "Select outcome to display:",
        ["Yes", "No", "Both"],
        horizontal=True
    )

    # Filter data based on outcome selection
    if outcome_filter == "Both":
        plot_data = data.copy()
    else:
        plot_data = data[data['outcome'] == outcome_filter].copy()

    # Create interactive plot
    fig = go.Figure()

    # Get unique resolution dates
    resolution_dates = sorted(plot_data['resolution_date'].unique())

    # Color palette
    colors = px.colors.qualitative.Plotly

    for i, res_date in enumerate(resolution_dates):
        for outcome in ['Yes', 'No']:
            if outcome_filter != "Both" and outcome != outcome_filter:
                continue

            subset = plot_data[
                (plot_data['resolution_date'] == res_date) &
                (plot_data['outcome'] == outcome)
            ].sort_values('date')

            if len(subset) == 0:
                continue

            line_style = 'solid' if outcome == 'Yes' else 'dash'
            color = colors[i % len(colors)]

            fig.add_trace(go.Scatter(
                x=subset['date'],
                y=subset['implied_rate'],
                mode='lines+markers',
                name=f"{res_date} ({outcome})",
                line=dict(dash=line_style, color=color),
                marker=dict(size=4, color=color),
                hovertemplate=(
                    f"<b>{res_date} ({outcome})</b><br>" +
                    "Date: %{x}<br>" +
                    "Implied Rate: %{y:.2%}<br>" +
                    "<extra></extra>"
                )
            ))

    fig.update_layout(
        title=f"Implied Rates for {selected_event_title}",
        xaxis_title="Date",
        yaxis_title="Implied Annual Rate",
        yaxis_tickformat=".0%",
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
    - Each line represents a different resolution date for the same underlying event
    - Solid lines = "Yes" outcome, Dashed lines = "No" outcome
    - Higher rates indicate lower market probabilities (more uncertain outcomes)
    - Rates should generally increase as resolution date approaches (due to decreasing time value)
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
