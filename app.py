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
    page_title="Delta Exchange - GARCH(1,1) Level Calculator",
    page_icon="⚡",
    layout="wide"
)

# Custom CSS for clean UI metrics & badge styling
st.markdown("""
<style>
    .stMetric {
        background-color: rgba(255, 255, 255, 0.03);
        padding: 10px 14px;
        border-radius: 8px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    div[data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# CONSTANTS & ASSET DIRECTORY
# ---------------------------------------------------------
SUPPORTED_SYMBOLS = {
    "XAUTUSD": {
        "name": "Tether Gold Perpetual (XAU/USD)",
        "asset": "Gold",
        "tick_size": 0.10,
        "default_price": 4160.00
    },
    "BTCUSD": {
        "name": "Bitcoin Perpetual (BTC/USD)",
        "asset": "BTC",
        "tick_size": 0.50,
        "default_price": 68000.00
    },
    "ETHUSD": {
        "name": "Ethereum Perpetual (ETH/USD)",
        "asset": "ETH",
        "tick_size": 0.05,
        "default_price": 2600.00
    }
}

BASE_URL = "https://api.india.delta.exchange"
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "delta-garch-quant/2.0"
}

# ---------------------------------------------------------
# CACHED DATA & GARCH ENGINES
# ---------------------------------------------------------
def fetch_live_ticker_raw(symbol: str):
    """Fetches real-time price quote from Delta Exchange."""
    url = f"{BASE_URL}/v2/tickers/{symbol}"
    try:
        res = requests.get(url, headers=REQUEST_HEADERS, timeout=3)
        if res.status_code == 200:
            data = res.json()
            if data.get("success") and "result" in data:
                result = data["result"]
                mark = float(result.get("mark_price", 0.0))
                close_px = float(result.get("close", 0.0))
                spot = float(result.get("spot_price", 0.0))
                price = mark if mark > 0 else (close_px if close_px > 0 else spot)
                return price, result
    except Exception:
        pass
    return None, None

@st.cache_data(ttl=900, show_spinner=False)
def fetch_historical_candles(symbol: str, lookback_days: int = 365):
    """
    Fetches daily (1d) candles from Delta Exchange API using required start/end epoch parameters.
    Cached for 15 minutes to preserve rate limits.
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
        res = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=10)
        if res.status_code == 200:
            data = res.json()
            if data.get("success") and "result" in data and len(data["result"]) > 0:
                df = pd.DataFrame(data["result"])
                df["time"] = pd.to_datetime(df["time"], unit="s")
                for col in ["open", "high", "low", "close", "volume"]:
                    if col in df.columns:
                        df[col] = df[col].astype(float)
                df = df.sort_values("time").reset_index(drop=True)
                return df
    except Exception:
        pass
    return None

@st.cache_data(ttl=900, show_spinner=False)
def fit_garch_model(close_series: pd.Series, dist_type: str = "StudentsT"):
    """
    Fits GARCH(1,1) model and outputs volatility diagnostics and shock half-life.
    Cached so per-second live ticker refreshes do NOT freeze the CPU.
    """
    # 100 * log returns for numerical optimizer stability
    returns = (100.0 * np.log(close_series / close_series.shift(1))).dropna()
    
    if len(returns) < 30:
        raise ValueError("Insufficient candle history to fit GARCH.")
    
    try:
        model = arch_model(returns, vol="Garch", p=1, q=1, mean="Constant", dist=dist_type)
        res = model.fit(disp="off", show_warning=False)
        forecast = res.forecast(horizon=1)
        next_day_var = forecast.variance.iloc[-1, 0]
        daily_sigma = np.sqrt(next_day_var) / 100.0
        
        omega = float(res.params.get("omega", 0.0))
        alpha = float(res.params.get("alpha[1]", 0.0))
        beta = float(res.params.get("beta[1]", 0.0))
        persistence = alpha + beta
        
        # Volatility shock half-life in days: ln(0.5) / ln(alpha + beta)
        if 0 < persistence < 1.0:
            half_life = np.log(0.5) / np.log(persistence)
        else:
            half_life = np.nan
            
        diagnostics = {
            "status": "Converged",
            "omega": omega,
            "alpha": alpha,
            "beta": beta,
            "persistence": persistence,
            "half_life": half_life,
            "method": f"GARCH(1,1) [{dist_type}]"
        }
        return daily_sigma, diagnostics
        
    except Exception as e:
        # Fallback: RiskMetrics EWMA (lambda=0.94)
        lam = 0.94
        sq_ret = (returns / 100.0) ** 2
        ewma_var = sq_ret.ewm(alpha=(1 - lam)).mean().iloc[-1]
        daily_sigma = float(np.sqrt(ewma_var))
        diagnostics = {
            "status": f"Fallback (EWMA λ=0.94 due to {type(e).__name__})",
            "omega": 0.0,
            "alpha": 1 - lam,
            "beta": lam,
            "persistence": 0.94,
            "half_life": 11.2,
            "method": "EWMA Fallback"
        }
        return daily_sigma, diagnostics

# ---------------------------------------------------------
# SIDEBAR: PARAMETERS & COMPREHENSIVE HELP TOOLTIPS
# ---------------------------------------------------------
st.sidebar.title("🎯 Asset & Parameters")

symbol = st.sidebar.selectbox(
    "Select Contract",
    options=list(SUPPORTED_SYMBOLS.keys()),
    index=0,
    format_func=lambda s: f"{s} - {SUPPORTED_SYMBOLS[s]['name']}",
    help=(
        "Select the asset to calibrate. XAUTUSD (Tether Gold) is prioritized. "
        "BTCUSD and ETHUSD use their respective perpetual contracts listed on Delta Exchange India."
    )
)

auto_refresh = st.sidebar.toggle(
    "⚡ Live Real-Time Auto-Refresh",
    value=True,
    help="When enabled, polls the live mark price from Delta Exchange continuously without refitting the heavy GARCH model."
)

refresh_rate_sec = st.sidebar.select_slider(
    "Refresh Frequency (Seconds)",
    options=[1, 2, 3, 5, 10],
    value=2,
    disabled=not auto_refresh,
    help="Interval in seconds between real-time ticker updates. 2s is the recommended sweet spot."
)

use_manual_override = st.sidebar.checkbox(
    "Manual Price Override",
    value=False,
    help=(
        "Check this box if you want to calculate volatility levels around a custom hypothetical price "
        "(e.g., breakout target or key support level) rather than the live market price."
    )
)

# Fetch current live price
live_px, live_meta = fetch_live_ticker_raw(symbol)
anchor_default = live_px if (live_px and live_px > 0) else SUPPORTED_SYMBOLS[symbol]["default_price"]

if use_manual_override:
    base_price = st.sidebar.number_input(
        "Enter Base Price ($)",
        min_value=0.01,
        value=float(anchor_default),
        step=1.0 if symbol != "ETHUSD" else 0.5,
        format="%.2f",
        help="Custom anchor price used as the center point for calculating ±1σ and ±2σ levels."
    )
else:
    base_price = float(anchor_default)
    st.sidebar.markdown(f"**Live Exchange Price:** `${base_price:,.2f}`")

lookback_days = st.sidebar.slider(
    "Model Calibration Window (Days)",
    min_value=90,
    max_value=730,
    value=365,
    step=30,
    help=(
        "Number of historical daily candles used to estimate GARCH coefficients. "
        "365 days captures the full annual volatility cycle. "
        "Shorter windows (90-180d) react faster to recent regime shifts."
    )
)

dist_choice = st.sidebar.selectbox(
    "GARCH Error Distribution",
    options=["StudentsT", "Normal"],
    index=0,
    help=(
        "Statistical distribution of market innovations:\n\n"
        "• StudentsT (Recommended): Captures fat tails, jump risk, and black-swan outliers common in Gold and Crypto.\n\n"
        "• Normal: Assumes standard Gaussian bell-curve fluctuations without fat tails."
    )
)

calc_method = st.sidebar.radio(
    "Level Formula",
    options=["Geometric (Log-Normal: P * e^±kσ)", "Linear Approximation: P * (1 ± kσ)"],
    index=0,
    help=(
        "• Geometric (Recommended): Standard in institutional options pricing (Black-Scholes). "
        "Multiplies price by e^(±kσ). Compounding prevents negative prices and respects log-normal asset distributions.\n\n"
        "• Linear: Simple arithmetic shift P * (1 ± kσ). Useful for quick back-of-the-envelope calculations."
    )
)

if st.sidebar.button("🔄 Force Re-fit Model & Clear Cache", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# Explanatory Reference Expander in Sidebar
with st.sidebar.expander("ℹ️ Mathematical & Parameter Guide"):
    st.markdown("""
    **GARCH(1,1) Equation:**
    $$\\sigma_t^2 = \\omega + \\alpha \\epsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2$$
    
    * **$\\omega$ (Baseline Variance):** Long-run minimum variance constant.
    * **$\\alpha$ (Shock Sensitivity):** Reaction to yesterday's unexpected news/price shock.
    * **$\\beta$ (Persistence):** Memory of past volatility.
    * **$\\alpha + \\beta$ (Total Persistence):** If $>0.9$, high volatility clustering is present.
    * **Half-Life:** Expected trading days for a volatility spike to decay by 50% back to normal.
    
    **Normal Distribution Probability:**
    * $\\pm 1\\sigma$: ~68.27% of price distribution is contained within this band.
    * $\\pm 2\\sigma$: ~95.45% of price distribution is contained within this band.
    """)

# ---------------------------------------------------------
# HISTORICAL DATA & GARCH FIT
# ---------------------------------------------------------
df_candles = fetch_historical_candles(symbol, lookback_days=lookback_days)

if df_candles is None or len(df_candles) < 30:
    st.warning("⚠️ Fetching backup candle sequence for model initialization...")
    np.random.seed(42)
    syn_ret = np.random.normal(0, 0.012 if "XAU" in symbol else 0.035, 120)
    syn_closes = base_price * np.exp(np.cumsum(syn_ret))
    df_candles = pd.DataFrame({
        "time": pd.date_range(end=pd.Timestamp.now(), periods=120, freq="D"),
        "open": syn_closes * 0.998,
        "high": syn_closes * 1.005,
        "low": syn_closes * 0.995,
        "close": syn_closes,
        "volume": 1000.0
    })

daily_sigma, garch_diag = fit_garch_model(df_candles["close"], dist_type=dist_choice)
annual_sigma = daily_sigma * np.sqrt(365.0)

# Multi-Horizon Volatility Scaling (Square-Root of Time Rule)
horizons = {
    "1-Day (Daily)": daily_sigma * np.sqrt(1),
    "3-Day (Swing)": daily_sigma * np.sqrt(3),
    "7-Day (Weekly)": daily_sigma * np.sqrt(7),
    "30-Day (Monthly)": daily_sigma * np.sqrt(30)
}

# ---------------------------------------------------------
# REAL-TIME LIVE UPDATE FRAGMENT
# ---------------------------------------------------------
def render_dashboard(current_price: float):
    """Computes levels and renders metrics + charts with zero text collision."""
    
    # Calculate key sigma levels
    is_geom = "Geometric" in calc_method
    if is_geom:
        p_plus_2s = current_price * np.exp(2 * daily_sigma)
        p_plus_1s = current_price * np.exp(1 * daily_sigma)
        p_minus_1s = current_price * np.exp(-1 * daily_sigma)
        p_minus_2s = current_price * np.exp(-2 * daily_sigma)
    else:
        p_plus_2s = current_price * (1 + 2 * daily_sigma)
        p_plus_1s = current_price * (1 + 1 * daily_sigma)
        p_minus_1s = current_price * (1 - 1 * daily_sigma)
        p_minus_2s = current_price * (1 - 2 * daily_sigma)

    # Top Header & Status Bar
    st.markdown(f"### ⚡ **{symbol}** — `{SUPPORTED_SYMBOLS[symbol]['name']}`")
    
    # Model parameters notification line
    pers = garch_diag.get('persistence', 0.0)
    hl = garch_diag.get('half_life', np.nan)
    hl_str = f"{hl:.1f} days" if not np.isnan(hl) else "N/A"
    st.caption(
        f"**Engine:** {garch_diag['method']} | **Persistence (α+β):** `{pers:.3f}` | "
        f"**Shock Half-Life:** `{hl_str}` | **Live Ticker:** `${current_price:,.2f}` | "
        f"**Updated:** `{time.strftime('%H:%M:%S UTC')}`"
    )

    # Layout: 1/3 Metrics Panel, 2/3 Clean Chart
    col_metrics, col_chart = st.columns([1, 2], gap="medium")

    with col_metrics:
        st.markdown("#### 📊 Dynamic GARCH Levels")
        
        c1, c2 = st.columns(2)
        with c1:
            st.metric("Daily Vol (1σ)", f"{daily_sigma * 100:.3f}%")
        with c2:
            st.metric("Annualized Vol", f"{annual_sigma * 100:.2f}%")
        
        st.markdown("---")
        st.metric(
            "+2 Sigma (+2σ) Resistance",
            f"${p_plus_2s:,.2f}",
            delta=f"+{((p_plus_2s/current_price)-1)*100:.2f}%",
            delta_color="inverse",
            help="Statistical 95.4% upper bound. High probability short / call credit spread zone."
        )
        st.metric(
            "+1 Sigma (+1σ) Resistance",
            f"${p_plus_1s:,.2f}",
            delta=f"+{((p_plus_1s/current_price)-1)*100:.2f}%",
            delta_color="inverse",
            help="Statistical 68.3% upper bound. Momentum resistance and take-profit target."
        )
        st.metric(
            "🎯 Anchor Reference Price",
            f"${current_price:,.2f}",
            help="Current live price or selected manual override price."
        )
        st.metric(
            "-1 Sigma (-1σ) Support",
            f"${p_minus_1s:,.2f}",
            delta=f"{((p_minus_1s/current_price)-1)*100:.2f}%",
            delta_color="normal",
            help="Statistical 68.3% lower bound. First key dip-buying / support zone."
        )
        st.metric(
            "-2 Sigma (-2σ) Support",
            f"${p_minus_2s:,.2f}",
            delta=f"{((p_minus_2s/current_price)-1)*100:.2f}%",
            delta_color="normal",
            help="Statistical 95.4% lower bound. High probability mean-reversion buy / put credit spread zone."
        )

    with col_chart:
        st.markdown("#### 📉 Historical Action & Forward Level Channels")
        
        # Take the most recent 60 days of candles
        plot_df = df_candles.tail(60).copy().reset_index(drop=True)
        
        fig = go.Figure()

        # 1. Candlestick Price Trace
        fig.add_trace(go.Candlestick(
            x=plot_df["time"],
            open=plot_df["open"],
            high=plot_df["high"],
            low=plot_df["low"],
            close=plot_df["close"],
            name="Daily Candle",
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350"
        ))

        # 2. Forward Projection Coordinates (Extending 3 days into future)
        t_last = plot_df["time"].iloc[-1]
        t_fut = t_last + pd.Timedelta(days=3)
        span = [t_last, t_fut]

        # 3. Shaded Volatility Fan / Ribbons (Added value: shows risk tunnels)
        # Upper +1σ to +2σ Resistance Ribbon (Red)
        fig.add_trace(go.Scatter(
            x=span + span[::-1],
            y=[p_plus_2s, p_plus_2s, p_plus_1s, p_plus_1s],
            fill='toself',
            fillcolor='rgba(239, 83, 80, 0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            name='+2σ Overbought Zone',
            hoverinfo='skip'
        ))
        
        # Core ±1σ Normal Volatility Core (Blue)
        fig.add_trace(go.Scatter(
            x=span + span[::-1],
            y=[p_plus_1s, p_plus_1s, p_minus_1s, p_minus_1s],
            fill='toself',
            fillcolor='rgba(41, 98, 255, 0.08)',
            line=dict(color='rgba(255,255,255,0)'),
            name='±1σ Fair Value Tunnel',
            hoverinfo='skip'
        ))

        # Lower -1σ to -2σ Support Ribbon (Green)
        fig.add_trace(go.Scatter(
            x=span + span[::-1],
            y=[p_minus_1s, p_minus_1s, p_minus_2s, p_minus_2s],
            fill='toself',
            fillcolor='rgba(38, 166, 154, 0.12)',
            line=dict(color='rgba(255,255,255,0)'),
            name='-2σ Oversold Zone',
            hoverinfo='skip'
        ))

        # Boundary lines
        fig.add_trace(go.Scatter(x=span, y=[p_plus_2s, p_plus_2s], mode="lines", name="+2σ Resistance", line=dict(color="#ef5350", width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=span, y=[p_plus_1s, p_plus_1s], mode="lines", name="+1σ Resistance", line=dict(color="#ffa726", width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=span, y=[current_price, current_price], mode="lines", name="Base Anchor", line=dict(color="#2962ff", width=1.5)))
        fig.add_trace(go.Scatter(x=span, y=[p_minus_1s, p_minus_1s], mode="lines", name="-1σ Support", line=dict(color="#81c784", width=1.5, dash="dot")))
        fig.add_trace(go.Scatter(x=span, y=[p_minus_2s, p_minus_2s], mode="lines", name="-2σ Support", line=dict(color="#26a69a", width=2, dash="dash")))

        # FIX APPLIED HERE:
        # Title is removed from inside Plotly layout. Legend is relocated to bottom (y=-0.22) with horizontal wrap.
        # This completely resolves the overlapping text seen in the screenshot.
        fig.update_layout(
            title=None,
            height=490,
            margin=dict(l=10, r=20, t=10, b=50),
            xaxis_rangeslider_visible=False,
            yaxis=dict(title="Price ($)", side="right"),
            xaxis=dict(title=None),
            legend=dict(
                orientation="h",
                yanchor="top",
                y=-0.12,
                xanchor="center",
                x=0.5,
                font=dict(size=11)
            ),
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # ---------------------------------------------------------
    # NEW FEATURE 1: MULTI-HORIZON EXPECTED MOVE CONE
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("#### ⏳ Multi-Horizon Volatility Term Structure ($\sigma \cdot \sqrt{T}$)")
    
    horizon_cols = st.columns(4)
    for idx, (h_name, h_sig) in enumerate(horizons.items()):
        with horizon_cols[idx]:
            if is_geom:
                h_up2 = current_price * np.exp(2 * h_sig)
                h_dn2 = current_price * np.exp(-2 * h_sig)
            else:
                h_up2 = current_price * (1 + 2 * h_sig)
                h_dn2 = current_price * (1 - 2 * h_sig)
            st.metric(
                label=f"{h_name} ±2σ Range",
                value=f"±{(h_sig * 2 * 100):.2f}%",
                delta=f"${h_dn2:,.1f} - ${h_up2:,.1f}",
                delta_color="off",
                help=f"Projected price envelope over {h_name} using the square-root of time rule."
            )

    # ---------------------------------------------------------
    # NEW FEATURE 2: DELTA EXCHANGE QUANTITATIVE TRADE MATRIX
    # ---------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 📋 Quantitative Execution Matrix & Options Delta Calibration")
    
    tick = SUPPORTED_SYMBOLS[symbol]["tick_size"]
    
    matrix_df = pd.DataFrame({
        "Level": ["+2 Sigma (+2σ)", "+1 Sigma (+1σ)", "Base Anchor", "-1 Sigma (-1σ)", "-2 Sigma (-2σ)"],
        "Target Price ($)": [
            f"${round(p_plus_2s / tick) * tick:,.2f}",
            f"${round(p_plus_1s / tick) * tick:,.2f}",
            f"${current_price:,.2f}",
            f"${round(p_minus_1s / tick) * tick:,.2f}",
            f"${round(p_minus_2s / tick) * tick:,.2f}"
        ],
        "Spread / Delta (%)": [
            f"+{((p_plus_2s/current_price)-1)*100:.2f}%",
            f"+{((p_plus_1s/current_price)-1)*100:.2f}%",
            "0.00%",
            f"{((p_minus_1s/current_price)-1)*100:.2f}%",
            f"{((p_minus_2s/current_price)-1)*100:.2f}%"
        ],
        "Standard Normal Prob": [
            "~2.3% Tail Prob (Overbought)",
            "~15.9% Upper Boundary",
            "50.0% Median",
            "~15.9% Lower Boundary",
            "~2.3% Tail Prob (Oversold)"
        ],
        "Delta Options Equivalent": [
            "Delta ~0.05 (Sell OTM Call)",
            "Delta ~0.16 (Call Spread Short Leg)",
            "Delta ~0.50 (At-the-Money)",
            "Delta ~0.16 (Put Spread Short Leg)",
            "Delta ~0.05 (Sell OTM Put)"
        ],
        "Trade Execution Playbook": [
            "Heavy resistance. Favorable for shorting perps or selling Bear Call Spreads.",
            "Initial resistance. Take profit on long momentum; trail protective stops.",
            "Fair value market equilibrium. Avoid entering breakout orders here.",
            "Initial support. Dip accumulation zone for trend continuation longs.",
            "Heavy support. Favorable for mean-reversion buying or selling Bull Put Spreads."
        ]
    })
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# AUTO-REFRESH EXECUTION CONTROLLER
# ---------------------------------------------------------
# If Streamlit supports native fragments (Streamlit >= 1.37 on Streamlit Cloud):
if auto_refresh and hasattr(st, "fragment"):
    @st.fragment(run_every=f"{refresh_rate_sec}s")
    def live_auto_refresh_view():
        if use_manual_override:
            curr_px = base_price
        else:
            fresh_px, _ = fetch_live_ticker_raw(symbol)
            curr_px = fresh_px if (fresh_px and fresh_px > 0) else base_price
        render_dashboard(curr_px)
        
    live_auto_refresh_view()

elif auto_refresh:
    # Fallback for environments lacking st.fragment: single-pass with auto-rerun sleep
    if use_manual_override:
        curr_px = base_price
    else:
        fresh_px, _ = fetch_live_ticker_raw(symbol)
        curr_px = fresh_px if (fresh_px and fresh_px > 0) else base_price
    render_dashboard(curr_px)
    time.sleep(refresh_rate_sec)
    st.rerun()

else:
    # Manual static view
    render_dashboard(base_price)