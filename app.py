import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
import plotly.graph_objects as go
from arch import arch_model

# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------
st.set_page_config(
    page_title="Delta Exchange - GARCH(1,1) Volatility Calculator",
    page_icon="⚡",
    layout="wide"
)

st.title("⚡ Delta Exchange India: GARCH(1,1) Level Calculator")
st.caption("Real-time daily $\pm 1\sigma$ and $\pm 2\sigma$ volatility boundaries calibrated on Delta Exchange historical candles.")

# Supported symbols with XAUTUSD prioritized
SUPPORTED_SYMBOLS = {
    "XAUTUSD": "Tether Gold Perpetual (Gold/USD)",
    "BTCUSD": "Bitcoin Perpetual (BTC/USD)",
    "ETHUSD": "Ethereum Perpetual (ETH/USD)"
}
BASE_URL = "https://api.india.delta.exchange"
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "delta-garch-calculator/1.0"
}

# ---------------------------------------------------------
# API DATA FETCHERS
# ---------------------------------------------------------
@st.cache_data(ttl=60)
def fetch_live_ticker(symbol: str):
    """Fetches real-time mark and spot prices from Delta Exchange."""
    url = f"{BASE_URL}/v2/tickers/{symbol}"
    try:
        res = requests.get(url, headers=REQUEST_HEADERS, timeout=8)
        if res.status_code == 200:
            data = res.json()
            if data.get("success") and "result" in data:
                result = data["result"]
                mark_price = float(result.get("mark_price", 0.0))
                close_price = float(result.get("close", 0.0))
                spot_price = float(result.get("spot_price", 0.0))
                # Choose best available live reference price
                live_price = mark_price if mark_price > 0 else (close_price if close_price > 0 else spot_price)
                return live_price, result
    except Exception as e:
        st.warning(f"Live ticker fetch warning: {e}")
    return None, None

@st.cache_data(ttl=1800)
def fetch_delta_daily_candles(symbol: str, lookback_days: int = 365):
    """
    Fetches daily (1d) historical candles from Delta Exchange API.
    Properly sends required 'start' and 'end' epoch second timestamps.
    """
    end_time = int(time.time())
    start_time = end_time - (lookback_days * 86400)
    
    url = f"{BASE_URL}/v2/history/candles"
    params = {
        "resolution": "1d",
        "symbol": symbol,
        "start": start_time,
        "end": end_time
    }
    
    try:
        response = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=12)
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and "result" in data and len(data["result"]) > 0:
                df = pd.DataFrame(data["result"])
                df["time"] = pd.to_datetime(df["time"], unit="s")
                numeric_cols = ["open", "high", "low", "close", "volume"]
                for col in numeric_cols:
                    if col in df.columns:
                        df[col] = df[col].astype(float)
                df = df.sort_values("time").reset_index(drop=True)
                return df
            else:
                st.error(f"Delta API returned no candle records for {symbol}: {data.get('error')}")
        else:
            st.error(f"Delta API Candle Error (HTTP {response.status_code}): {response.text}")
    except Exception as err:
        st.error(f"Candle API Connection Error: {err}")
    return None

# ---------------------------------------------------------
# GARCH(1,1) ESTIMATION ENGINE
# ---------------------------------------------------------
def estimate_garch_volatility(close_series: pd.Series, dist_type: str = "StudentsT"):
    """
    Fits GARCH(1,1) model with heavy-tail protection and automated fallback.
    Returns:
        daily_sigma (float): daily standard deviation in decimal form (e.g. 0.018 for 1.8%)
        annualized_vol (float): annualized volatility in decimal form
        fit_summary (str): textual model status/parameters
    """
    # Calculate percentage log returns (scaled by 100 for stable optimizer gradients)
    returns = (100.0 * np.log(close_series / close_series.shift(1))).dropna()
    
    if len(returns) < 30:
        raise ValueError("Insufficient history (need >= 30 clean returns).")
    
    try:
        # Student's t accounts for fat tails in Gold and Crypto assets
        model = arch_model(
            returns,
            vol="Garch",
            p=1,
            q=1,
            mean="Constant",
            dist=dist_type
        )
        res = model.fit(disp="off", show_warning=False)
        forecast = res.forecast(horizon=1)
        next_day_var = forecast.variance.iloc[-1, 0]
        daily_sigma = np.sqrt(next_day_var) / 100.0
        annualized_sigma = daily_sigma * np.sqrt(365.0)
        
        omega = res.params.get("omega", np.nan)
        alpha = res.params.get("alpha[1]", np.nan)
        beta = res.params.get("beta[1]", np.nan)
        status = f"GARCH(1,1) Converged | ω: {omega:.4f}, α: {alpha:.4f}, β: {beta:.4f} (Persistence: {alpha + beta:.3f})"
        return daily_sigma, annualized_sigma, status
    
    except Exception as garch_err:
        # Robust Fallback: RiskMetrics Exponentially Weighted Moving Average (EWMA, lambda=0.94)
        lam = 0.94
        sq_ret = (returns / 100.0) ** 2
        ewma_var = sq_ret.ewm(alpha=(1 - lam)).mean().iloc[-1]
        daily_sigma = np.sqrt(ewma_var)
        annualized_sigma = daily_sigma * np.sqrt(365.0)
        status = f"Optimizer Warning ({garch_err}); Fallback to EWMA (λ=0.94)."
        return daily_sigma, annualized_sigma, status

# ---------------------------------------------------------
# SIDEBAR CONTROLS
# ---------------------------------------------------------
st.sidebar.header("🎯 Asset & Parameters")

# Priority on XAUTUSD as first option
symbol = st.sidebar.selectbox(
    "Select Contract",
    options=list(SUPPORTED_SYMBOLS.keys()),
    index=0,
    format_func=lambda s: f"{s} - {SUPPORTED_SYMBOLS[s]}"
)

# Fetch live reference price from Delta
live_price, ticker_raw = fetch_live_ticker(symbol)
default_price = float(live_price) if live_price is not None and live_price > 0 else (4164.00 if symbol == "XAUTUSD" else 65000.0)

use_custom_price = st.sidebar.checkbox("Manual Price Override", value=False)
if use_custom_price:
    base_price = st.sidebar.number_input(
        "Enter Base Reference Price ($)",
        min_value=0.01,
        value=default_price,
        step=1.0 if symbol != "ETHUSD" else 0.5,
        format="%.2f"
    )
else:
    base_price = default_price
    st.sidebar.markdown(f"**Current Live Price:** `${base_price:,.2f}`")

lookback_days = st.sidebar.slider("Model Calibration Window (Days)", min_value=90, max_value=730, value=365, step=30)
dist_choice = st.sidebar.selectbox("GARCH Error Distribution", ["StudentsT", "Normal"], index=0, help="Student's t captures heavy tail risk common in crypto and metals.")
calc_method = st.sidebar.radio("Level Formula", ["Geometric (Log-Normal: P * e^±kσ)", "Linear Approximation: P * (1 ± kσ)"])

if st.sidebar.button("🔄 Refresh Market Data"):
    st.cache_data.clear()
    st.rerun()

# ---------------------------------------------------------
# COMPUTATION WORKFLOW
# ---------------------------------------------------------
with st.spinner(f"Connecting to Delta Exchange India & fetching candles for {symbol}..."):
    df_candles = fetch_delta_daily_candles(symbol, lookback_days=lookback_days)

if df_candles is None or len(df_candles) < 30:
    st.warning("⚠️ Insufficient historical candles received from Delta Exchange API. Bootstrapping fallback distribution...")
    np.random.seed(42)
    daily_returns_sim = np.random.normal(0, 0.012 if "XAU" in symbol else 0.035, 120)
    fake_prices = base_price * np.exp(np.cumsum(daily_returns_sim))
    df_candles = pd.DataFrame({
        "time": pd.date_range(end=pd.Timestamp.now(), periods=120, freq="D"),
        "close": fake_prices,
        "high": fake_prices * 1.005,
        "low": fake_prices * 0.995,
        "open": fake_prices * 0.998
    })

# Run GARCH model
daily_sigma, annual_sigma, model_status = estimate_garch_volatility(df_candles["close"], dist_type=dist_choice)

# Calculate boundaries
if "Geometric" in calc_method:
    plus_2sigma = base_price * np.exp(2 * daily_sigma)
    plus_1sigma = base_price * np.exp(1 * daily_sigma)
    minus_1sigma = base_price * np.exp(-1 * daily_sigma)
    minus_2sigma = base_price * np.exp(-2 * daily_sigma)
else:
    plus_2sigma = base_price * (1 + 2 * daily_sigma)
    plus_1sigma = base_price * (1 + 1 * daily_sigma)
    minus_1sigma = base_price * (1 - 1 * daily_sigma)
    minus_2sigma = base_price * (1 - 2 * daily_sigma)

# ---------------------------------------------------------
# DASHBOARD DISPLAY
# ---------------------------------------------------------
col_metrics, col_chart = st.columns([1, 2])

with col_metrics:
    st.subheader(f"📊 {symbol} Volatility Levels")
    st.caption(model_status)
    
    st.metric("Daily Volatility ($\sigma$)", f"{daily_sigma * 100:.3f}%", help="Forecast 1-day standard deviation from GARCH(1,1).")
    st.metric("Annualized Volatility", f"{annual_sigma * 100:.2f}%", help="Daily volatility multiplied by sqrt(365).")
    
    st.markdown("---")
    st.metric("+2 Sigma (+2σ) Resistance", f"${plus_2sigma:,.2f}", f"+{((plus_2sigma/base_price)-1)*100:.2f}%")
    st.metric("+1 Sigma (+1σ) Resistance", f"${plus_1sigma:,.2f}", f"+{((plus_1sigma/base_price)-1)*100:.2f}%")
    st.metric("🎯 Base Anchor Price", f"${base_price:,.2f}")
    st.metric("-1 Sigma (-1σ) Support", f"${minus_1sigma:,.2f}", f"{((minus_1sigma/base_price)-1)*100:.2f}%")
    st.metric("-2 Sigma (-2σ) Support", f"${minus_2sigma:,.2f}", f"{((minus_2sigma/base_price)-1)*100:.2f}%")

with col_chart:
    st.subheader("📉 Historical Action & Forward Level Channels")
    
    # Real historical candles (last 60 periods for clean visual display)
    plot_df = df_candles.tail(60).copy().reset_index(drop=True)
    
    fig = go.Figure()
    
    # Candlestick or line chart
    if "open" in plot_df.columns and "high" in plot_df.columns:
        fig.add_trace(go.Candlestick(
            x=plot_df["time"],
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="Daily Price Action",
            increasing_line_color="#26a69a", decreasing_line_color="#ef5350"
        ))
    else:
        fig.add_trace(go.Scatter(
            x=plot_df["time"], y=plot_df["close"],
            mode="lines", name="Close Price", line=dict(color="#2962FF", width=2)
        ))
    
    # Forward projection lines
    last_time = plot_df["time"].iloc[-1]
    future_time = last_time + pd.Timedelta(days=2)
    time_span = [last_time, future_time]
    
    fig.add_trace(go.Scatter(x=time_span, y=[plus_2sigma, plus_2sigma], mode="lines", name="+2σ (95.4%)", line=dict(color="red", width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=time_span, y=[plus_1sigma, plus_1sigma], mode="lines", name="+1σ (68.3%)", line=dict(color="orange", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=time_span, y=[base_price, base_price], mode="lines", name="Base Anchor", line=dict(color="blue", width=1.5)))
    fig.add_trace(go.Scatter(x=time_span, y=[minus_1sigma, minus_1sigma], mode="lines", name="-1σ Support", line=dict(color="lightgreen", width=1.5, dash="dot")))
    fig.add_trace(go.Scatter(x=time_span, y=[minus_2sigma, minus_2sigma], mode="lines", name="-2σ Support", line=dict(color="green", width=2, dash="dash")))
    
    fig.update_layout(
        title=f"{symbol} Delta Exchange Market Levels & GARCH Dynamic Band",
        yaxis_title="Price ($)",
        xaxis_title="Date",
        height=480,
        margin=dict(l=20, r=20, t=40, b=20),
        xaxis_rangeslider_visible=False,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# LEVEL SUMMARY TABLE
# ---------------------------------------------------------
st.markdown("---")
st.subheader("📋 Quantitative Summary & Trade Plan Matrix")

summary_table = pd.DataFrame({
    "Sigma Level": ["+2 Sigma (+2σ)", "+1 Sigma (+1σ)", "Base Anchor", "-1 Sigma (-1σ)", "-2 Sigma (-2σ)"],
    "Price Target ($)": [f"${plus_2sigma:,.2f}", f"${plus_1sigma:,.2f}", f"${base_price:,.2f}", f"${minus_1sigma:,.2f}", f"${minus_2sigma:,.2f}"],
    "Expected Deviation": [
        f"+{((plus_2sigma/base_price)-1)*100:.2f}%",
        f"+{((plus_1sigma/base_price)-1)*100:.2f}%",
        "0.00%",
        f"{((minus_1sigma/base_price)-1)*100:.2f}%",
        f"{((minus_2sigma/base_price)-1)*100:.2f}%"
    ],
    "Statistical Probability": [
        "~2.3% Tail Prob (Overextended)",
        "~15.9% Probability Boundary",
        "Median Expectation",
        "~15.9% Probability Boundary",
        "~2.3% Tail Prob (Oversold)"
    ],
    "Trading Interpretation": [
        "Major Resistance / High probability short or call credit spread",
        "Intermediate Resistance / Take-profit on long momentum",
        "Current Market Mark / Settlement Anchor",
        "Intermediate Support / Dip buying zone for trend continuation",
        "Major Support / High probability mean-reversion long / put credit spread"
    ]
})

st.dataframe(summary_table, use_container_width=True)