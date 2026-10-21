import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timezone, timedelta
from arch import arch_model

# ---------------------------------------------------------
# PAGE SETUP & TIMEZONE (IST = UTC+05:30)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Delta Exchange India - Ultra Quant GARCH Terminal",
    page_icon="⚡",
    layout="wide"
)

IST = timezone(timedelta(hours=5, minutes=30))

# Custom High-Contrast Terminal Styling
st.markdown("""
<style>
    /* Global Container */
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    /* Metrics Box Defaults */
    div[data-testid="stMetric"] {
        background: rgba(22, 27, 34, 0.85);
        border: 1px solid rgba(48, 54, 61, 0.9);
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }
    
    /* Level Cards */
    .level-card {
        border-radius: 8px;
        padding: 16px 22px;
        margin-bottom: 11px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-left: 8px solid;
        transition: transform 0.12s ease-in-out, filter 0.12s ease-in-out;
    }
    .level-card:hover {
        transform: translateY(-1px);
        filter: brightness(1.06);
    }
    
    .level-p3 { background: rgba(239, 83, 80, 0.14); border-left-color: #d32f2f; border: 1px solid rgba(239, 83, 80, 0.40); }
    .level-p2 { background: rgba(239, 83, 80, 0.08); border-left-color: #ef5350; border: 1px solid rgba(239, 83, 80, 0.28); }
    .level-p1 { background: rgba(255, 167, 38, 0.08); border-left-color: #ffa726; border: 1px solid rgba(255, 167, 38, 0.28); }
    .level-base { background: rgba(41, 182, 246, 0.13); border-left-color: #29b6f6; border: 1px solid rgba(41, 182, 246, 0.42); }
    .level-m1 { background: rgba(102, 187, 106, 0.08); border-left-color: #66bb6a; border: 1px solid rgba(102, 187, 106, 0.28); }
    .level-m2 { background: rgba(38, 166, 154, 0.08); border-left-color: #26a69a; border: 1px solid rgba(38, 166, 154, 0.28); }
    .level-m3 { background: rgba(0, 150, 136, 0.14); border-left-color: #00796b; border: 1px solid rgba(0, 150, 136, 0.40); }

    .level-name {
        font-size: 1.05rem;
        font-weight: 700;
        letter-spacing: 0.4px;
    }
    .level-desc {
        font-size: 0.81rem;
        opacity: 0.82;
        margin-top: 3px;
    }
    .level-price {
        font-size: 1.48rem;
        font-weight: 800;
        font-family: 'SF Mono', 'Roboto Mono', Menlo, Consolas, monospace;
        text-align: right;
    }
    .level-spread {
        font-size: 0.86rem;
        font-weight: 600;
        text-align: right;
    }

    .stat-pill {
        display: inline-block;
        padding: 2px 7px;
        border-radius: 4px;
        font-size: 0.72rem;
        font-weight: 600;
        margin-left: 6px;
    }

    .bracket-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 15px 18px;
        margin-top: 6px;
        line-height: 1.75;
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
        "strike_interval": 10.0,
        "step": 0.50,
        "contract_size": "0.001 oz",
        "default_price": 4160.00
    },
    "BTCUSD": {
        "name": "Bitcoin Perpetual (BTC/USD)",
        "asset": "BTC",
        "tick_size": 0.50,
        "strike_interval": 500.0,
        "step": 10.00,
        "contract_size": "0.001 BTC",
        "default_price": 68000.00
    },
    "ETHUSD": {
        "name": "Ethereum Perpetual (ETH/USD)",
        "asset": "ETH",
        "tick_size": 0.05,
        "strike_interval": 25.0,
        "step": 1.00,
        "contract_size": "0.01 ETH",
        "default_price": 2600.00
    }
}

BASE_URL = "https://api.india.delta.exchange"
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "delta-garch-quant-terminal/5.0"
}

# ---------------------------------------------------------
# MARKET DATA RETRIEVERS & ROBUST GARCH ESTIMATORS
# ---------------------------------------------------------
def fetch_live_ticker(sym: str):
    """Fetches real-time price & 24h market stats from Delta Exchange India."""
    url = f"{BASE_URL}/v2/tickers/{sym}"
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
def fetch_historical_candles(sym: str, lookback_days: int = 365):
    """Fetches daily (1d) candles from Delta Exchange API using start/end epoch second bounds."""
    end_time = int(time.time())
    start_time = end_time - (lookback_days * 86400)
    
    url = f"{BASE_URL}/v2/history/candles"
    params = {
        "resolution": "1d",
        "symbol": sym,
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
def fit_advanced_garch(close_series: pd.Series, dist_type: str = "StudentsT", is_asymmetric: bool = False):
    """
    Fits GARCH(1,1) or Asymmetric GJR-GARCH(1,1) with Student's-t or Normal error distribution.
    Cached for 15 minutes to guarantee sub-second frontend response.
    """
    returns = (100.0 * np.log(close_series / close_series.shift(1))).dropna()
    
    if len(returns) < 30:
        raise ValueError("Insufficient candle history to fit GARCH.")
    
    # 30-Day Historical Realized Volatility for Volatility Risk Spread
    rv_30d = float(returns.tail(30).std() * np.sqrt(365.0) / 100.0)
    
    try:
        o_param = 1 if is_asymmetric else 0
        model = arch_model(returns, vol="Garch", p=1, o=o_param, q=1, mean="Constant", dist=dist_type)
        res = model.fit(disp="off", show_warning=False)
        forecast = res.forecast(horizon=1)
        next_day_var = forecast.variance.iloc[-1, 0]
        daily_sigma = np.sqrt(next_day_var) / 100.0
        
        omega = float(res.params.get("omega", 0.0))
        alpha = float(res.params.get("alpha[1]", 0.0))
        gamma = float(res.params.get("gamma[1]", 0.0)) if is_asymmetric else 0.0
        beta = float(res.params.get("beta[1]", 0.0))
        
        persistence = alpha + beta + (0.5 * gamma if is_asymmetric else 0.0)
        
        if 0.0 < persistence < 0.9999 and omega > 0:
            uncond_var = (omega / (1.0 - persistence)) / 10000.0
            uncond_ann_vol = np.sqrt(max(0.0, uncond_var)) * np.sqrt(365.0)
            half_life = np.log(0.5) / np.log(persistence)
        else:
            uncond_ann_vol = daily_sigma * np.sqrt(365.0)
            half_life = np.nan
            
        diagnostics = {
            "status": "Converged",
            "omega": omega,
            "alpha": alpha,
            "gamma": gamma,
            "beta": beta,
            "persistence": persistence,
            "half_life": half_life,
            "uncond_vol": uncond_ann_vol,
            "rv_30d": rv_30d,
            "method": f"GJR-GARCH(1,1) [{dist_type}]" if is_asymmetric else f"Standard GARCH(1,1) [{dist_type}]"
        }
        return daily_sigma, diagnostics
        
    except Exception as e:
        lam = 0.94
        sq_ret = (returns / 100.0) ** 2
        ewma_var = sq_ret.ewm(alpha=(1 - lam)).mean().iloc[-1]
        daily_sigma = float(np.sqrt(ewma_var))
        diagnostics = {
            "status": f"Fallback (EWMA λ=0.94: {type(e).__name__})",
            "omega": 0.0,
            "alpha": 1 - lam,
            "gamma": 0.0,
            "beta": lam,
            "persistence": 0.94,
            "half_life": 11.2,
            "uncond_vol": daily_sigma * np.sqrt(365.0),
            "rv_30d": rv_30d,
            "method": "EWMA Fallback (λ=0.94)"
        }
        return daily_sigma, diagnostics

# ---------------------------------------------------------
# SIDEBAR CONTROLS & SYNCHRONIZED PLACEHOLDER
# ---------------------------------------------------------
st.sidebar.title("⚡ Quant Terminal Controls")

# Symbol Selector (XAUTUSD Prioritized)
symbol = st.sidebar.selectbox(
    "Select Contract",
    options=list(SUPPORTED_SYMBOLS.keys()),
    index=0,
    format_func=lambda s: f"{s} - {SUPPORTED_SYMBOLS[s]['name']}",
    help=(
        "Select derivative contract on Delta Exchange India:\n\n"
        "• XAUTUSD: Tether Gold Token Perpetual. Prioritized asset; settled in USD.\n"
        "• BTCUSD: Bitcoin Perpetual Futures.\n"
        "• ETHUSD: Ethereum Perpetual Futures."
    )
)

auto_refresh = st.sidebar.toggle(
    "⚡ Live Real-Time Auto-Refresh",
    value=True,
    help="When enabled, streams live market ticks from Delta Exchange India continuously without reloading the full page."
)

refresh_rate_sec = st.sidebar.select_slider(
    "Refresh Frequency (Seconds)",
    options=[1, 2, 3, 5, 10],
    value=2,
    disabled=not auto_refresh,
    help="Polling frequency in seconds. 2s offers low-latency live quotes without hitting exchange rate limits (20,000 quota/5min)."
)

st.sidebar.markdown("---")

# Dynamic Synchronized Live Price Box in Sidebar
sidebar_price_slot = st.sidebar.empty()

# ---------------------------------------------------------
# MANUAL PRICE OVERRIDE SYSTEM (ISOLATED PER ASSET)
# ---------------------------------------------------------
use_manual_override = st.sidebar.checkbox(
    "Manual Price Override",
    value=False,
    key=f"override_flag_{symbol}",
    help="Check this box to anchor volatility calculations around a hypothetical price (e.g. breakout trigger or key pivot) rather than the live exchange mark."
)

live_px, live_meta = fetch_live_ticker(symbol)
market_price_anchor = live_px if (live_px and live_px > 0) else SUPPORTED_SYMBOLS[symbol]["default_price"]

# Isolated Session State variable
manual_input_key = f"manual_num_val_{symbol}"
if manual_input_key not in st.session_state:
    st.session_state[manual_input_key] = float(market_price_anchor)

if use_manual_override:
    c_in, c_rst = st.sidebar.columns([3, 1])
    with c_rst:
        st.write("")
        st.write("")
        if st.button("↺ Reset", key=f"btn_reset_{symbol}", help="Reset manual price box back to live market price"):
            st.session_state[manual_input_key] = float(market_price_anchor)
            st.rerun()

    with c_in:
        base_price = st.number_input(
            "Enter Base Price ($)",
            min_value=0.01,
            max_value=10_000_000.0,
            step=SUPPORTED_SYMBOLS[symbol]["step"],
            format="%.2f",
            key=manual_input_key,
            help=f"Custom base price for {symbol}. Step size: ${SUPPORTED_SYMBOLS[symbol]['step']:.2f}."
        )
else:
    base_price = float(market_price_anchor)

lookback_days = st.sidebar.slider(
    "Model Calibration Window (Days)",
    min_value=90,
    max_value=730,
    value=365,
    step=30,
    help=(
        "Number of historical daily candles used to fit the GARCH parameters:\n\n"
        "• 365 Days (Recommended): Full annual volatility regime cycle.\n"
        "• 90-180 Days: Faster response to recent volatility breakouts or contractions."
    )
)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Model Specification")

model_type = st.sidebar.selectbox(
    "GARCH Model Type",
    options=["Standard GARCH(1,1)", "Asymmetric GJR-GARCH(1,1)"],
    index=1 if symbol != "XAUTUSD" else 0,
    help=(
        "• Standard GARCH(1,1): Symmetric variance reaction to both up and down moves.\n\n"
        "• Asymmetric GJR-GARCH(1,1): Captures the empirical 'leverage effect' where down moves trigger heavier panic volatility than up moves."
    )
)
is_asymmetric = "GJR" in model_type

dist_choice = st.sidebar.selectbox(
    "Error Distribution",
    options=["StudentsT", "Normal"],
    index=0,
    help=(
        "Probability density for return residuals:\n\n"
        "• StudentsT (Recommended): Heavy fat-tails; captures crypto and gold flash crashes.\n\n"
        "• Normal: Classic Gaussian distribution."
    )
)

calc_method = st.sidebar.radio(
    "Level Formula",
    options=["Geometric (Log-Normal: P * e^±kσ)", "Linear Approximation: P * (1 ± kσ)"],
    index=0,
    help=(
        "• Geometric (Recommended): Standard in Black-Scholes options pricing. Compounding preserves positive non-zero boundaries.\n\n"
        "• Linear Approximation: Simple arithmetic offset P * (1 ± kσ)."
    )
)

show_tail_levels = st.sidebar.checkbox(
    "Show Extreme ±3σ Black Swan Levels",
    value=False,
    help="Includes ±1.5σ and ±3.0σ extreme tail boundaries in the execution cards."
)

if st.sidebar.button("🔄 Force Re-fit GARCH Model", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# Mathematical Guide Expander in Sidebar
with st.sidebar.expander("ℹ️ Mathematical & Parameter Guide"):
    st.markdown("""
    **GJR-GARCH Volatility Formula:**
    $$\\sigma_t^2 = \\omega + (\\alpha + \\gamma I_{t-1}) \\epsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2$$
    
    * **$\\omega$ (Baseline Variance):** Long-run minimum variance constant.
    * **$\\alpha$ (Shock Sensitivity):** Reaction to yesterday's unexpected news shock.
    * **$\\gamma$ (Asymmetry / Leverage):** Additional variance applied when yesterday was a down day ($I_{t-1}=1$).
    * **$\\beta$ (Persistence):** Memory of past volatility clustering.
    * **Persistence ($P$):** $\\alpha + \\beta + \\frac{\\gamma}{2}$. If $P > 0.9$, volatility exhibits persistent clustering.
    * **Shock Half-Life:** Trading days required for a volatility spike to decay by 50% back to baseline.
    
    **Normal Distribution Probability Bounds:**
    * **$\\pm 1\\sigma$:** 68.27% of return outcomes (Touch Prob: ~31.73%).
    * **$\\pm 2\\sigma$:** 95.45% of return outcomes (Touch Prob: ~4.55%).
    * **$\\pm 3\\sigma$:** 99.73% of return outcomes (Touch Prob: ~0.27%).
    """)

# ---------------------------------------------------------
# HISTORICAL DATA & GARCH CALIBRATION
# ---------------------------------------------------------
df_candles = fetch_historical_candles(symbol, lookback_days=lookback_days)

if df_candles is None or len(df_candles) < 30:
    st.warning("⚠️ Initializing synthetic fallback candles...")
    np.random.seed(42)
    syn_ret = np.random.normal(0, 0.012 if "XAU" in symbol else 0.035, 120)
    syn_closes = base_price * np.exp(np.cumsum(syn_ret))
    df_candles = pd.DataFrame({
        "time": pd.date_range(end=pd.Timestamp.now(), periods=120, freq="D"),
        "close": syn_closes
    })

daily_sigma, garch_diag = fit_advanced_garch(
    df_candles["close"],
    dist_type=dist_choice,
    is_asymmetric=is_asymmetric
)
annual_sigma = daily_sigma * np.sqrt(365.0)

# Multi-Horizon Scaling (Square-Root of Time Rule)
horizons = {
    "1-Day (Intraday)": daily_sigma * np.sqrt(1),
    "3-Day (Swing)": daily_sigma * np.sqrt(3),
    "7-Day (Weekly Expiry)": daily_sigma * np.sqrt(7),
    "14-Day (Bi-Weekly)": daily_sigma * np.sqrt(14),
    "30-Day (Monthly Expiry)": daily_sigma * np.sqrt(30)
}

# ---------------------------------------------------------
# CLEAN DASHBOARD RENDERER
# ---------------------------------------------------------
def render_clean_dashboard(current_price: float, live_market_price: float):
    """Renders clean, chart-free, institutional cards with synchronized IST clock."""
    
    # Synchronize sidebar indicator
    if use_manual_override:
        sidebar_price_slot.markdown(f"""
        <div style="background:rgba(41,182,246,0.12); border:1px solid rgba(41,182,246,0.35); padding:10px 14px; border-radius:8px; margin-bottom:12px;">
            <div style="font-size:0.75rem; color:#29b6f6; font-weight:700;">MANUAL OVERRIDE ACTIVE</div>
            <div style="font-size:1.25rem; font-weight:800; font-family:monospace; color:#29b6f6;">${current_price:,.2f}</div>
            <div style="font-size:0.75rem; opacity:0.7; margin-top:3px;">Live Exchange Mark: ${live_market_price:,.2f}</div>
        </div>
        """, unsafe_allow_html=True)
    else:
        sidebar_price_slot.markdown(f"""
        <div style="background:rgba(0,230,118,0.08); border:1px solid rgba(0,230,118,0.25); padding:10px 14px; border-radius:8px; margin-bottom:12px;">
            <div style="font-size:0.75rem; color:#00e676; font-weight:700;">LIVE EXCHANGE PRICE (IST)</div>
            <div style="font-size:1.25rem; font-weight:800; font-family:monospace; color:#00e676;">${current_price:,.2f}</div>
            <div style="font-size:0.75rem; opacity:0.7; margin-top:3px;">Delta Exchange India Feed</div>
        </div>
        """, unsafe_allow_html=True)

    # 1. Level Projections
    is_geom = "Geometric" in calc_method
    if is_geom:
        p_plus_3s = current_price * np.exp(3 * daily_sigma)
        p_plus_2_5s = current_price * np.exp(2.5 * daily_sigma)
        p_plus_2s = current_price * np.exp(2 * daily_sigma)
        p_plus_1_5s = current_price * np.exp(1.5 * daily_sigma)
        p_plus_1s = current_price * np.exp(1 * daily_sigma)
        p_minus_1s = current_price * np.exp(-1 * daily_sigma)
        p_minus_1_5s = current_price * np.exp(-1.5 * daily_sigma)
        p_minus_2s = current_price * np.exp(-2 * daily_sigma)
        p_minus_2_5s = current_price * np.exp(-2.5 * daily_sigma)
        p_minus_3s = current_price * np.exp(-3 * daily_sigma)
    else:
        p_plus_3s = current_price * (1 + 3 * daily_sigma)
        p_plus_2_5s = current_price * (1 + 2.5 * daily_sigma)
        p_plus_2s = current_price * (1 + 2 * daily_sigma)
        p_plus_1_5s = current_price * (1 + 1.5 * daily_sigma)
        p_plus_1s = current_price * (1 + 1 * daily_sigma)
        p_minus_1s = current_price * (1 - 1 * daily_sigma)
        p_minus_1_5s = current_price * (1 - 1.5 * daily_sigma)
        p_minus_2s = current_price * (1 - 2 * daily_sigma)
        p_minus_2_5s = current_price * (1 - 2.5 * daily_sigma)
        p_minus_3s = current_price * (1 - 3 * daily_sigma)

    tick = SUPPORTED_SYMBOLS[symbol]["tick_size"]
    s_int = SUPPORTED_SYMBOLS[symbol]["strike_interval"]
    
    # Delta Exchange Strike Snap Function
    def snap_strike(px):
        return round(px / s_int) * s_int

    p_plus_3s_rnd = round(p_plus_3s / tick) * tick
    p_plus_2s_rnd = round(p_plus_2s / tick) * tick
    p_plus_1s_rnd = round(p_plus_1s / tick) * tick
    p_minus_1s_rnd = round(p_minus_1s / tick) * tick
    p_minus_2s_rnd = round(p_minus_2s / tick) * tick
    p_minus_3s_rnd = round(p_minus_3s / tick) * tick

    # Parametric Risk Metrics
    var_95_pct = 1.645 * daily_sigma
    var_99_pct = 2.326 * daily_sigma
    cvar_95_pct = 2.063 * daily_sigma
    var_95_usd = current_price * var_95_pct

    # Volatility Spread (Forecast vs 30d Realized)
    rv_30d = garch_diag.get("rv_30d", annual_sigma)
    vol_spread = (annual_sigma - rv_30d) * 100.0

    # Volatility Regime Thermometer
    uncond_vol = garch_diag.get("uncond_vol", annual_sigma)
    vol_ratio = annual_sigma / uncond_vol if uncond_vol > 0 else 1.0
    if vol_ratio < 0.85:
        regime_badge = "🟢 Compressed Vol (Breakout Pending)"
    elif vol_ratio <= 1.15:
        regime_badge = "🔵 Equilibrium Vol (Normal)"
    elif vol_ratio <= 1.50:
        regime_badge = "🟡 Elevated Vol (Clustering Active)"
    else:
        regime_badge = "🔴 Extreme Vol (High Tail Risk)"

    # Safe Institutional Leverage Limit
    safe_max_leverage = max(1.0, min(50.0, 1.0 / (3.0 * annual_sigma)))

    # Accurate IST Time
    ist_now = datetime.now(IST)
    ist_time_str = ist_now.strftime("%H:%M:%S IST")

    # Header Row
    h_col1, h_col2 = st.columns([2.6, 1.4])
    with h_col1:
        st.title(f"⚡ {symbol} — Quantitative Volatility Terminal")
        pers = garch_diag.get('persistence', 0.0)
        hl = garch_diag.get('half_life', np.nan)
        hl_str = f"{hl:.1f} days" if not np.isnan(hl) else "N/A"
        gamma_str = f" | **Leverage (γ):** `{garch_diag.get('gamma', 0.0):.4f}`" if is_asymmetric else ""
        st.caption(
            f"**Contract:** {SUPPORTED_SYMBOLS[symbol]['name']} | "
            f"**Model:** {garch_diag['method']} | "
            f"**Persistence (P):** `{pers:.3f}` | "
            f"**Half-Life:** `{hl_str}`{gamma_str}"
        )
    with h_col2:
        st.metric(
            label="Live Delta Reference Price",
            value=f"${current_price:,.2f}",
            delta=f"Live Feed @ {ist_time_str}",
            delta_color="off"
        )

    # 24h Delta Exchange Live Market Stats Strip
    if live_meta:
        s1, s2, s3, s4, s5 = st.columns(5)
        with s1:
            ltp_chg = float(live_meta.get("ltp_change_24h", 0.0)) * 100
            st.metric("24h Price Change", f"{ltp_chg:+.2f}%", delta=f"{ltp_chg:+.2f}%")
        with s2:
            st.metric("24h High", f"${float(live_meta.get('high', 0.0)):,.2f}")
        with s3:
            st.metric("24h Low", f"${float(live_meta.get('low', 0.0)):,.2f}")
        with s4:
            st.metric("24h Volume (Turnover)", f"${float(live_meta.get('turnover_usd', 0.0)):,.0f}")
        with s5:
            mark_p = float(live_meta.get("mark_price", 0.0))
            spot_p = float(live_meta.get("spot_price", 0.0))
            basis = mark_p - spot_p
            st.metric("Mark-Spot Basis", f"${basis:+.2f}", delta=f"{basis:+.2f}")

    st.markdown("---")

    # Risk Metrics & Volatility Diagnostics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    with m1:
        st.metric(
            "Forecast Volatility (1σ)",
            f"{daily_sigma * 100:.3f}%",
            delta=f"Annual: {annual_sigma * 100:.2f}%",
            delta_color="off",
            help="Forecast 1-day standard deviation from the GARCH model."
        )
    with m2:
        st.metric(
            "Vol Spread (Forecast vs RV30)",
            f"{vol_spread:+.2f}%",
            delta="Premium Expanding" if vol_spread > 0 else "Compressing",
            delta_color="normal" if vol_spread > 0 else "inverse",
            help="Annualized GARCH Forecast Volatility minus 30-Day Historical Realized Volatility."
        )
    with m3:
        st.metric(
            "1-Day 95% VaR (Risk)",
            f"${var_95_usd:,.2f}",
            delta=f"-{var_95_pct * 100:.2f}% Max 1D Loss",
            delta_color="inverse",
            help="Value at Risk: 95% of trading days experience losses within this limit."
        )
    with m4:
        st.metric(
            "1-Day 95% Expected Shortfall",
            f"-{cvar_95_pct * 100:.2f}%",
            delta=f"-${(current_price * cvar_95_pct):,.2f} CVaR",
            delta_color="inverse",
            help="Conditional VaR (Expected Shortfall): Expected loss conditional on breaching the 95% VaR boundary."
        )
    with m5:
        st.metric(
            "Prudent Max Leverage",
            f"{safe_max_leverage:.1f}x",
            delta=regime_badge.split(" ")[0] + " " + regime_badge.split(" ")[1],
            delta_color="off",
            help="Institutional risk cap: Maximum leverage that keeps a 2.5σ tail shock within maintenance margin."
        )

    st.markdown("---")

    # Primary GARCH Level Cards
    st.markdown("### 🎯 GARCH Volatility Level Cards")

    if show_tail_levels:
        # +3 Sigma Card
        st.markdown(f"""
        <div class="level-card level-p3">
            <div>
                <div class="level-name" style="color: #ef5350;">
                    🔴 +3 Sigma (+3σ) Extreme Resistance 
                    <span class="stat-pill" style="background: rgba(239,83,80,0.3); color: #ef5350;">99.73% Bound</span>
                    <span class="stat-pill" style="background: rgba(255,255,255,0.08); color: #ddd;">Touch Prob: ~0.27%</span>
                    <span class="stat-pill" style="background: rgba(255,255,255,0.08); color: #ffd54f;">Strike: ${snap_strike(p_plus_3s_rnd):,.0f}</span>
                </div>
                <div class="level-desc">Extreme Black Swan Spike • Circuit Breaker Threshold • Liquidity Vacuum Zone</div>
            </div>
            <div>
                <div class="level-price" style="color: #ef5350;">${p_plus_3s_rnd:,.2f}</div>
                <div class="level-spread" style="color: #ef5350;">+{((p_plus_3s/current_price)-1)*100:.2f}% from anchor</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # +2 Sigma Card
    st.markdown(f"""
    <div class="level-card level-p2">
        <div>
            <div class="level-name" style="color: #ef5350;">
                🔴 +2 Sigma (+2σ) Resistance 
                <span class="stat-pill" style="background: rgba(239,83,80,0.2); color: #ef5350;">95.45% Bound</span>
                <span class="stat-pill" style="background: rgba(255,255,255,0.08); color: #ddd;">Touch Prob: ~4.55%</span>
                <span class="stat-pill" style="background: rgba(255,255,255,0.08); color: #ffd54f;">Strike: ${snap_strike(p_plus_2s_rnd):,.0f}</span>
            </div>
            <div class="level-desc">Statistical Envelope Cap • Extreme Overbought • Optimal for Selling OTM Calls / Bear Call Spreads</div>
        </div>
        <div>
            <div class="level-price" style="color: #ef5350;">${p_plus_2s_rnd:,.2f}</div>
            <div class="level-spread" style="color: #ef5350;">+{((p_plus_2s/current_price)-1)*100:.2f}% from anchor</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # +1 Sigma Card
    st.markdown(f"""
    <div class="level-card level-p1">
        <div>
            <div class="level-name" style="color: #ffa726;">
                🟠 +1 Sigma (+1σ) Resistance 
                <span class="stat-pill" style="background: rgba(255,167,38,0.2); color: #ffa726;">68.27% Bound</span>
                <span class="stat-pill" style="background: rgba(255,255,255,0.08); color: #ddd;">Touch Prob: ~31.73%</span>
                <span class="stat-pill" style="background: rgba(255,255,255,0.08); color: #ffd54f;">Strike: ${snap_strike(p_plus_1s_rnd):,.0f}</span>
            </div>
            <div class="level-desc">Momentum Barrier • Take-Profit Target for Longs • First Resistance Zone</div>
        </div>
        <div>
            <div class="level-price" style="color: #ffa726;">${p_plus_1s_rnd:,.2f}</div>
            <div class="level-spread" style="color: #ffa726;">+{((p_plus_1s/current_price)-1)*100:.2f}% from anchor</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Base Anchor Card
    st.markdown(f"""
    <div class="level-card level-base">
        <div>
            <div class="level-name" style="color: #29b6f6;">
                🔵 🎯 Base Anchor Reference 
                <span class="stat-pill" style="background: rgba(41,182,246,0.25); color: #29b6f6;">{"Manual Override" if use_manual_override else "Live Mark"}</span>
            </div>
            <div class="level-desc">Market Equilibrium Point • Baseline Delta Exchange Execution Anchor</div>
        </div>
        <div>
            <div class="level-price" style="color: #29b6f6;">${current_price:,.2f}</div>
            <div class="level-spread" style="color: #29b6f6;">0.00% Baseline</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -1 Sigma Card
    st.markdown(f"""
    <div class="level-card level-m1">
        <div>
            <div class="level-name" style="color: #66bb6a;">
                🟢 -1 Sigma (-1σ) Support 
                <span class="stat-pill" style="background: rgba(102,187,106,0.2); color: #66bb6a;">68.27% Bound</span>
                <span class="stat-pill" style="background: rgba(255,255,255,0.08); color: #ddd;">Touch Prob: ~31.73%</span>
                <span class="stat-pill" style="background: rgba(255,255,255,0.08); color: #ffd54f;">Strike: ${snap_strike(p_minus_1s_rnd):,.0f}</span>
            </div>
            <div class="level-desc">Trend Pullback Support • First Dip-Buying Accumulation Zone</div>
        </div>
        <div>
            <div class="level-price" style="color: #66bb6a;">${p_minus_1s_rnd:,.2f}</div>
            <div class="level-spread" style="color: #66bb6a;">{((p_minus_1s/current_price)-1)*100:.2f}% from anchor</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -2 Sigma Card
    st.markdown(f"""
    <div class="level-card level-m2">
        <div>
            <div class="level-name" style="color: #26a69a;">
                🟢 -2 Sigma (-2σ) Support 
                <span class="stat-pill" style="background: rgba(38,166,154,0.2); color: #26a69a;">95.45% Bound</span>
                <span class="stat-pill" style="background: rgba(255,255,255,0.08); color: #ddd;">Touch Prob: ~4.55%</span>
                <span class="stat-pill" style="background: rgba(255,255,255,0.08); color: #ffd54f;">Strike: ${snap_strike(p_minus_2s_rnd):,.0f}</span>
            </div>
            <div class="level-desc">Statistical Envelope Floor • Extreme Oversold • Put Spread Short Leg / Mean-Reversion Long</div>
        </div>
        <div>
            <div class="level-price" style="color: #26a69a;">${p_minus_2s_rnd:,.2f}</div>
            <div class="level-spread" style="color: #26a69a;">{((p_minus_2s/current_price)-1)*100:.2f}% from anchor</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if show_tail_levels:
        # -3 Sigma Card
        st.markdown(f"""
        <div class="level-card level-m3">
            <div>
                <div class="level-name" style="color: #00897b;">
                    🟢 -3 Sigma (-3σ) Extreme Support 
                    <span class="stat-pill" style="background: rgba(0,137,123,0.3); color: #4db6ac;">99.73% Bound</span>
                    <span class="stat-pill" style="background: rgba(255,255,255,0.08); color: #ddd;">Touch Prob: ~0.27%</span>
                    <span class="stat-pill" style="background: rgba(255,255,255,0.08); color: #ffd54f;">Strike: ${snap_strike(p_minus_3s_rnd):,.0f}</span>
                </div>
                <div class="level-desc">Flash Crash Floor • Deep Institutional Liquidation Exhaustion Boundary</div>
            </div>
            <div>
                <div class="level-price" style="color: #00897b;">${p_minus_3s_rnd:,.2f}</div>
                <div class="level-spread" style="color: #00897b;">{((p_minus_3s/current_price)-1)*100:.2f}% from anchor</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Multi-Horizon Volatility Envelopes Row
    st.markdown("### ⏳ Multi-Horizon Volatility Term Structure ($\sigma \cdot \sqrt{T}$)")
    
    h_cols = st.columns(len(horizons))
    for idx, (h_name, h_sig) in enumerate(horizons.items()):
        with h_cols[idx]:
            if is_geom:
                h_up2 = current_price * np.exp(2 * h_sig)
                h_dn2 = current_price * np.exp(-2 * h_sig)
            else:
                h_up2 = current_price * (1 + 2 * h_sig)
                h_dn2 = current_price * (1 - 2 * h_sig)
            
            st.metric(
                label=h_name,
                value=f"±{(h_sig * 2 * 100):.2f}%",
                delta=f"${round(h_dn2/tick)*tick:,.1f} — ${round(h_up2/tick)*tick:,.1f}",
                delta_color="off",
                help=f"Projected ±2σ (95.4%) expected price dispersion over {h_name}."
            )

    st.markdown("---")

    # Delta Exchange Order Bracket Generator
    st.markdown("### 🛡️ Delta Exchange Order Bracket Generator")
    st.caption("Quantitative parameters matching Delta Exchange India Bracket Order API structure.")
    
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.markdown("**🟢 Bullish Mean-Reversion Plan (Long at -1σ / -2σ)**")
        entry_long = p_minus_1s_rnd
        tp_long = current_price
        sl_long = round(p_minus_2_5s / tick) * tick
        risk_long = max(tick, entry_long - sl_long)
        reward_long = max(tick, tp_long - entry_long)
        rr_long = reward_long / risk_long
        
        st.markdown(f"""
        <div class="bracket-box">
            <b>Target Entry Price:</b> <code>${entry_long:,.2f}</code> (-1σ Support)<br>
            <b>Take-Profit Trigger:</b> <code>${tp_long:,.2f}</code> (Anchor Baseline)<br>
            <b>Stop-Loss Trigger:</b> <code>${sl_long:,.2f}</code> (-2.5σ Tail Defense)<br>
            <b>Risk/Reward Ratio:</b> <span style="color:#66bb6a; font-weight:700;">1 : {rr_long:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

    with b_col2:
        st.markdown("**🔴 Bearish Mean-Reversion Plan (Short at +1σ / +2σ)**")
        entry_short = p_plus_1s_rnd
        tp_short = current_price
        sl_short = round(p_plus_2_5s / tick) * tick
        risk_short = max(tick, sl_short - entry_short)
        reward_short = max(tick, entry_short - tp_short)
        rr_short = reward_short / risk_short
        
        st.markdown(f"""
        <div class="bracket-box">
            <b>Target Entry Price:</b> <code>${entry_short:,.2f}</code> (+1σ Resistance)<br>
            <b>Take-Profit Trigger:</b> <code>${tp_short:,.2f}</code> (Anchor Baseline)<br>
            <b>Stop-Loss Trigger:</b> <code>${sl_short:,.2f}</code> (+2.5σ Tail Defense)<br>
            <b>Risk/Reward Ratio:</b> <span style="color:#ef5350; font-weight:700;">1 : {rr_short:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

    # Copyable Delta API Bracket Order Payload
    with st.expander("💻 Delta Exchange REST API Order Payload (JSON)"):
        bracket_json = f"""{{
  "product_symbol": "{symbol}",
  "order_type": "limit_order",
  "side": "buy",
  "size": 1,
  "limit_price": "{entry_long:.2f}",
  "bracket_take_profit_price": "{tp_long:.2f}",
  "bracket_stop_loss_price": "{sl_long:.2f}",
  "bracket_stop_trigger_method": "last_traded_price",
  "time_in_force": "gtc"
}}"""
        st.code(bracket_json, language="json")

    st.markdown("---")

    # Quantitative Execution Matrix Table
    st.markdown("### 📋 Delta Exchange Execution Matrix & Options Calibration")
    
    matrix_df = pd.DataFrame({
        "Level": ["+2 Sigma (+2σ)", "+1 Sigma (+1σ)", "Base Anchor", "-1 Sigma (-1σ)", "-2 Sigma (-2σ)"],
        "Target Price ($)": [
            f"${p_plus_2s_rnd:,.2f}",
            f"${p_plus_1s_rnd:,.2f}",
            f"${current_price:,.2f}",
            f"${p_minus_1s_rnd:,.2f}",
            f"${p_minus_2s_rnd:,.2f}"
        ],
        "Spread / Delta (%)": [
            f"+{((p_plus_2s/current_price)-1)*100:.2f}%",
            f"+{((p_plus_1s/current_price)-1)*100:.2f}%",
            "0.00%",
            f"{((p_minus_1s/current_price)-1)*100:.2f}%",
            f"{((p_minus_2s/current_price)-1)*100:.2f}%"
        ],
        "Terminal Expiry Prob": [
            "~2.28% Above",
            "~15.87% Above",
            "50.00% Baseline",
            "~15.87% Below",
            "~2.28% Below"
        ],
        "Intraday Touch Prob": [
            "~4.55% Touch",
            "~31.73% Touch",
            "100.00% Active",
            "~31.73% Touch",
            "~4.55% Touch"
        ],
        "Delta Options Strike Snap": [
            f"${snap_strike(p_plus_2s_rnd):,.0f} Call",
            f"${snap_strike(p_plus_1s_rnd):,.0f} Call",
            f"${snap_strike(current_price):,.0f} ATM",
            f"${snap_strike(p_minus_1s_rnd):,.0f} Put",
            f"${snap_strike(p_minus_2s_rnd):,.0f} Put"
        ],
        "Delta Options Calibration": [
            "Delta ~0.05 (Sell OTM Call / Credit Spread)",
            "Delta ~0.16 (Call Spread Short Leg)",
            "Delta ~0.50 (At-the-Money)",
            "Delta ~0.16 (Put Spread Short Leg)",
            "Delta ~0.05 (Sell OTM Put / Credit Spread)"
        ],
        "Execution Playbook": [
            "Major resistance. Favorable for shorting perps or selling Bear Call Spreads.",
            "Initial resistance. Take profit on long momentum; trail protective stops.",
            "Fair value market equilibrium. Avoid entering breakout orders here.",
            "Initial support. Dip accumulation zone for trend continuation longs.",
            "Major support. Favorable for mean-reversion buying or selling Bull Put Spreads."
        ]
    })
    st.dataframe(matrix_df, use_container_width=True, hide_index=True)


# ---------------------------------------------------------
# AUTO-REFRESH CONTROLLER (LIVE TICKER SYNC)
# ---------------------------------------------------------
if auto_refresh and hasattr(st, "fragment"):
    @st.fragment(run_every=f"{refresh_rate_sec}s")
    def live_auto_refresh_view():
        fresh_px, _ = fetch_live_ticker(symbol)
        live_mkt = fresh_px if (fresh_px and fresh_px > 0) else base_price
        curr_px = float(st.session_state[manual_input_key]) if use_manual_override else live_mkt
        render_clean_dashboard(curr_px, live_mkt)
        
    live_auto_refresh_view()

elif auto_refresh:
    fresh_px, _ = fetch_live_ticker(symbol)
    live_mkt = fresh_px if (fresh_px and fresh_px > 0) else base_price
    curr_px = float(st.session_state[manual_input_key]) if use_manual_override else live_mkt
    render_clean_dashboard(curr_px, live_mkt)
    time.sleep(refresh_rate_sec)
    st.rerun()

else:
    render_clean_dashboard(base_price, market_price_anchor)