import streamlit as st
import pandas as pd
import numpy as np
import requests
import plotly.graph_objects as go
from arch import arch_model

# Page Configuration
st.set_page_config(
    page_title="Delta Exchange - GARCH Level Calculator",
    page_icon="📈",
    layout="wide"
)

st.title("⚡ Delta Exchange India: GARCH(1,1) Level Calculator")
st.markdown("Automated daily **1-Sigma ($\pm 1\sigma$)** and **2-Sigma ($\pm 2\sigma$)** level calculation for **XAU/USDT**.")

# ---------------------------------------------------------
# SIDEBAR INPUTS
# ---------------------------------------------------------
st.sidebar.header("🔑 Dashboard Inputs")

# 1. Delta Credentials (Optional for Public Ticker/OHLC Data, included for Authenticated Endpoints)
api_key = st.sidebar.text_input("Delta API Key", type="password", help="Optional for fetching market OHLC data")
api_secret = st.sidebar.text_input("Delta API Secret", type="password", help="Optional for fetching market OHLC data")

# 2. Base Price Input
st.sidebar.markdown("---")
st.sidebar.subheader("🎯 Price Setting")
price_input = st.sidebar.number_input(
    "Enter Current / Target Price ($):",
    min_value=0.0,
    value=2650.00,
    step=0.50,
    format="%.2f"
)

# Symbol selection
symbol = st.sidebar.selectbox("Symbol", ["XAUTUSD", "XAUUSDT"], index=0)

# ---------------------------------------------------------
# HELPER FUNCTIONS (AUTOMATIC DATA & GARCH MODEL)
# ---------------------------------------------------------
@st.cache_data(ttl=3600)
def fetch_delta_daily_candles(symbol_name="XAUTUSD"):
    """Fetches historical daily candle data automatically from Delta Exchange India API."""
    url = f"https://api.india.delta.exchange/v2/history/candles?resolution=1d&symbol={symbol_name}"
    headers = {"Accept": "application/json"}
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        data = response.json()
        
        if data.get("success") and "result" in data:
            df = pd.DataFrame(data["result"])
            # Delta returns 'time', 'open', 'high', 'low', 'close', 'volume'
            df['close'] = df['close'].astype(float)
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df = df.sort_values('time').reset_index(drop=True)
            return df
        else:
            # Fallback mock data generation if endpoint format differs
            return None
    except Exception as e:
        st.error(f"Error fetching data from Delta Exchange: {e}")
        return None

def compute_garch_volatility(close_prices):
    """Calculates daily standard deviation (sigma) using GARCH(1,1)."""
    # Compute log returns (%)
    returns = 100 * np.log(close_prices / close_prices.shift(1)).dropna()
    
    # Fit GARCH(1,1) model: sigma_t^2 = omega + alpha * e_{t-1}^2 + beta * sigma_{t-1}^2
    model = arch_model(returns, vol='Garch', p=1, q=1, mean='Zero', dist='normal')
    fitted_model = model.fit(disp='off')
    
    # Forecast 1-step ahead variance
    forecast = fitted_model.forecast(horizon=1)
    predicted_variance = forecast.variance.iloc[-1, 0]
    
    # Convert forecasted percentage standard deviation back to decimal
    sigma_daily = np.sqrt(predicted_variance) / 100.0
    return sigma_daily

# ---------------------------------------------------------
# AUTOMATIC COMPUTATION ENGINE
# ---------------------------------------------------------
with st.spinner("Fetching market historical data & fitting GARCH(1,1) model..."):
    df_candles = fetch_delta_daily_candles(symbol)

if df_candles is None or len(df_candles) < 30:
    st.warning("⚠️ Could not fetch sufficient live candles from Delta API. Using historical simulated fallback data for GARCH fitting.")
    # Fallback log normal distribution simulation for backup
    np.random.seed(42)
    fake_closes = price_input * np.exp(np.cumsum(np.random.normal(0, 0.012, 100)))
    df_candles = pd.DataFrame({"close": fake_closes})

# Compute Sigma automatically
sigma_daily = compute_garch_volatility(df_candles['close'])

# Compute 1-Sigma and 2-Sigma levels
plus_2sigma = price_input * (1 + 2 * sigma_daily)
plus_1sigma = price_input * (1 + 1 * sigma_daily)
minus_1sigma = price_input * (1 - 1 * sigma_daily)
minus_2sigma = price_input * (1 - 2 * sigma_daily)

# ---------------------------------------------------------
# DISPLAY DASHBOARD RESULTS
# ---------------------------------------------------------
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("📊 GARCH Level Metrics")
    st.info(f"**Predicted Daily Volatility (Daily Sigma):** {sigma_daily * 100:.3f}%")
    
    st.metric(label="+2 Sigma (+2σ) Resistance", value=f"${plus_2sigma:,.2f}", delta=f"+{(2 * sigma_daily * 100):.2f}%")
    st.metric(label="+1 Sigma (+1σ) Resistance", value=f"${plus_1sigma:,.2f}", delta=f"+{(1 * sigma_daily * 100):.2f}%")
    st.metric(label="🎯 Input Base Price", value=f"${price_input:,.2f}")
    st.metric(label="-1 Sigma (-1σ) Support", value=f"${minus_1sigma:,.2f}", delta=f"-{(1 * sigma_daily * 100):.2f}%")
    st.metric(label="-2 Sigma (-2σ) Support", value=f"${minus_2sigma:,.2f}", delta=f"-{(2 * sigma_daily * 100):.2f}%")

with col_right:
    st.subheader("📉 Level Visualizer")
    
    fig = go.Figure()

    # Price levels lines
    fig.add_hline(y=plus_2sigma, line_dash="dash", line_color="red", annotation_text="+2σ Resistance", annotation_position="top right")
    fig.add_hline(y=plus_1sigma, line_dash="dot", line_color="orange", annotation_text="+1σ Resistance", annotation_position="top right")
    fig.add_hline(y=price_input, line_color="blue", annotation_text="Base Price", annotation_position="top right")
    fig.add_hline(y=minus_1sigma, line_dash="dot", line_color="lightgreen", annotation_text="-1σ Support", annotation_position="bottom right")
    fig.add_hline(y=minus_2sigma, line_dash="dash", line_color="green", annotation_text="-2σ Support", annotation_position="bottom right")

    fig.update_layout(
        title=f"XAU Volatility Channel around ${price_input:,.2f}",
        yaxis_title="Price ($)",
        xaxis_title="Visualization Range",
        height=420,
        showlegend=False
    )
    
    st.plotly_chart(fig, use_container_width=True)

# Detailed Breakdown Table
st.markdown("---")
st.subheader("📋 Summary Table")
summary_df = pd.DataFrame({
    "Level Name": ["+2 Sigma (+2σ)", "+1 Sigma (+1σ)", "Base Price", "-1 Sigma (-1σ)", "-2 Sigma (-2σ)"],
    "Price Level ($)": [plus_2sigma, plus_1sigma, price_input, minus_1sigma, minus_2sigma],
    "Percentage Deviation": [f"+{2*sigma_daily*100:.2f}%", f"+{1*sigma_daily*100:.2f}%", "0.00%", f"-{1*sigma_daily*100:.2f}%", f"-{2*sigma_daily*100:.2f}%"],
    "Market Meaning": ["Strong Overbought / Resistance", "Overbought / Resistance", "Entered Reference Price", "Oversold / Support", "Strong Oversold / Support"]
})
st.dataframe(summary_df, use_container_width=True)





# python3 -m venv garch
# source garch/bin/activate
# pip install streamlit pandas numpy arch requests plotly
# streamlit run app.py
