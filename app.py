import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timezone, timedelta
from arch import arch_model

# ---------------------------------------------------------
# PAGE CONFIGURATION & TIMEZONE
# ---------------------------------------------------------
st.set_page_config(
    page_title="Delta Exchange India - Advanced GARCH(1,1) Terminal",
    page_icon="⚡",
    layout="wide"
)

# Indian Standard Time (IST = UTC + 5:30)
IST = timezone(timedelta(hours=5, minutes=30))

# Custom Dark Terminal Theme CSS
st.markdown("""
<style>
    /* Metric & Card Container Defaults */
    .metric-card {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 10px;
        padding: 14px 18px;
        margin-bottom: 10px;
    }
    
    /* GARCH Level Cards Styling */
    .level-card {
        border-radius: 10px;
        padding: 15px 22px;
        margin-bottom: 11px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-left: 7px solid;
        transition: transform 0.15s ease-in-out;
    }
    .level-card:hover {
        filter: brightness(1.08);
    }
    
    .level-p2 {
        background: rgba(239, 83, 80, 0.09);
        border-left-color: #ef5350;
        border: 1px solid rgba(239, 83, 80, 0.28);
    }
    .level-p1 {
        background: rgba(255, 167, 38, 0.09);
        border-left-color: #ffa726;
        border: 1px solid rgba(255, 167, 38, 0.28);
    }
    .level-base {
        background: rgba(41, 182, 246, 0.14);
        border-left-color: #29b6f6;
        border: 1px solid rgba(41, 182, 246, 0.40);
    }
    .level-m1 {
        background: rgba(102, 187, 106, 0.09);
        border-left-color: #66bb6a;
        border: 1px solid rgba(102, 187, 106, 0.28);
    }
    .level-m2 {
        background: rgba(38, 166, 154, 0.09);
        border-left-color: #26a69a;
        border: 1px solid rgba(38, 166, 154, 0.28);
    }

    .level-name {
        font-size: 1.08rem;
        font-weight: 700;
        letter-spacing: 0.4px;
    }
    .level-desc {
        font-size: 0.82rem;
        opacity: 0.78;
        margin-top: 3px;
    }
    .level-price {
        font-size: 1.45rem;
        font-weight: 800;
        font-family: monospace;
        text-align: right;
    }
    .level-spread {
        font-size: 0.88rem;
        font-weight: 600;
        text-align: right;
    }

    /* Bracket order code box */
    .bracket-box {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 8px;
        padding: 12px 16px;
        margin-top: 8px;
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
        "step": 0.50,
        "default_price": 4160.00
    },
    "BTCUSD": {
        "name": "Bitcoin Perpetual (BTC/USD)",
        "asset": "BTC",
        "tick_size": 0.50,
        "step": 10.00,
        "default_price": 68000.00
    },
    "ETHUSD": {
        "name": "Ethereum Perpetual (ETH/USD)",
        "asset": "ETH",
        "tick_size": 0.05,
        "step": 1.00,
        "default_price": 2600.00
    }
}

BASE_URL = "https://api.india.delta.exchange"
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "delta-garch-quant-terminal/3.0"
}

# ---------------------------------------------------------
# DATA FETCHERS & GARCH ENGINE
# ---------------------------------------------------------
def fetch_live_ticker(sym: str):
    """Fetches real-time market price from Delta Exchange India."""
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
    """
    Fetches daily (1d) candles from Delta Exchange API using start/end epoch timestamps.
    Cached for 15 minutes to preserve network bandwidth and rate limits.
    """
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
    Fits GARCH(1,1) or Asymmetric GJR-GARCH(1,1) model and outputs volatility diagnostics,
    unconditional variance, and shock half-life.
    Cached to ensure per-second live ticker refreshes do NOT freeze the CPU.
    """
    returns = (100.0 * np.log(close_series / close_series.shift(1))).dropna()
    
    if len(returns) < 30:
        raise ValueError("Insufficient candle history to fit GARCH.")
    
    try:
        # If asymmetric is selected, o=1 enables GJR-GARCH leverage term
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
        
        # Effective persistence
        persistence = alpha + beta + (0.5 * gamma if is_asymmetric else 0.0)
        
        # Long-run unconditional annual volatility
        if 0 < persistence < 1.0 and omega > 0:
            uncond_var = (omega / (1.0 - persistence)) / 10000.0
            uncond_ann_vol = np.sqrt(uncond_var) * np.sqrt(365.0)
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
            "method": f"GJR-GARCH(1,1) [{dist_type}]" if is_asymmetric else f"Standard GARCH(1,1) [{dist_type}]"
        }
        return daily_sigma, diagnostics
        
    except Exception as e:
        # Robust Fallback: RiskMetrics EWMA (lambda=0.94)
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
            "method": "EWMA Fallback (λ=0.94)"
        }
        return daily_sigma, diagnostics

# ---------------------------------------------------------
# SIDEBAR CONTROLS & COMPREHENSIVE HELP TOOLTIPS
# ---------------------------------------------------------
st.sidebar.title("⚡ Quant Terminal Controls")

# Symbol Selection (XAUTUSD prioritized)
symbol = st.sidebar.selectbox(
    "Select Contract",
    options=list(SUPPORTED_SYMBOLS.keys()),
    index=0,
    format_func=lambda s: f"{s} - {SUPPORTED_SYMBOLS[s]['name']}",
    help=(
        "Select the asset on Delta Exchange India:\n\n"
        "• XAUTUSD: Tether Gold Perpetual (1 contract = 0.001 oz Gold, USD quote).\n"
        "• BTCUSD: Bitcoin Perpetual (1 contract = 0.001 BTC).\n"
        "• ETHUSD: Ethereum Perpetual (1 contract = 0.01 ETH)."
    )
)

auto_refresh = st.sidebar.toggle(
    "⚡ Live Real-Time Auto-Refresh",
    value=True,
    help="Continuously polls Delta Exchange India for live ticker quotes without re-estimating GARCH."
)

refresh_rate_sec = st.sidebar.select_slider(
    "Refresh Frequency (Seconds)",
    options=[1, 2, 3, 5, 10],
    value=2,
    disabled=not auto_refresh,
    help="Polling interval in seconds. 2s offers live tracking without triggering Delta Exchange rate limits."
)

# ---------------------------------------------------------
# ROBUST MANUAL PRICE OVERRIDE ENGINE (FIX FOR ETH / ALL)
# ---------------------------------------------------------
use_manual_override = st.sidebar.checkbox(
    "Manual Price Override",
    value=False,
    key=f"override_toggle_{symbol}",
    help="Anchor volatility channels around a custom price (e.g. key breakout level or custom entry) rather than the live market price."
)

# Fetch latest price
live_px, live_meta = fetch_live_ticker(symbol)
market_price_anchor = live_px if (live_px and live_px > 0) else SUPPORTED_SYMBOLS[symbol]["default_price"]

# Isolated session key per symbol avoids cross-asset state corruption
session_px_key = f"active_manual_px_{symbol}"
if session_px_key not in st.session_state:
    st.session_state[session_px_key] = float(market_price_anchor)

if use_manual_override:
    col_input, col_reset = st.sidebar.columns([3, 1])
    with col_reset:
        st.write("")
        st.write("")
        if st.button("↺ Sync", help="Reset manual price to live market mark"):
            st.session_state[session_px_key] = float(market_price_anchor)
            st.rerun()
            
    with col_input:
        base_price = st.number_input(
            "Enter Base Price ($)",
            min_value=0.01,
            max_value=10_000_000.0,
            value=float(st.session_state[session_px_key]),
            step=SUPPORTED_SYMBOLS[symbol]["step"],
            format="%.2f",
            key=f"widget_num_px_{symbol}",
            help=f"Custom anchor price for {symbol}. Step size: ${SUPPORTED_SYMBOLS[symbol]['step']:.2f}."
        )
        st.session_state[session_px_key] = base_price
else:
    base_price = float(market_price_anchor)
    st.sidebar.markdown(f"**Live Exchange Price:** `${base_price:,.2f}`")

lookback_days = st.sidebar.slider(
    "Model Calibration Window (Days)",
    min_value=90,
    max_value=730,
    value=365,
    step=30,
    help=(
        "Number of historical daily candles used to estimate GARCH parameters:\n\n"
        "• 365 Days: Optimal for capturing standard annual volatility regimes.\n"
        "• 90-180 Days: High responsiveness to recent volatility expansions or contractions."
    )
)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 Model Specification")

model_type = st.sidebar.selectbox(
    "GARCH Model Type",
    options=["Standard GARCH(1,1)", "Asymmetric GJR-GARCH(1,1)"],
    index=1 if symbol != "XAUTUSD" else 0,
    help=(
        "• Standard GARCH(1,1): Symmetric reaction to both positive and negative returns.\n\n"
        "• Asymmetric GJR-GARCH: Models the 'leverage effect' where down moves trigger higher volatility spikes than up moves."
    )
)
is_asymmetric = "GJR" in model_type

dist_choice = st.sidebar.selectbox(
    "Error Distribution",
    options=["StudentsT", "Normal"],
    index=0,
    help=(
        "Statistical innovation density:\n\n"
        "• StudentsT (Recommended): Captures heavy fat tails and jump kurtosis common in Gold and Crypto.\n\n"
        "• Normal: Standard Gaussian curve without tail clustering."
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

if st.sidebar.button("🔄 Force Re-fit GARCH Model", use_container_width=True):
    st.cache_data.clear()
    st.rerun()

# Comprehensive Math Expander in Sidebar
with st.sidebar.expander("ℹ️ Mathematical & Parameter Guide"):
    st.markdown("""
    **GJR-GARCH Equation:**
    $$\\sigma_t^2 = \\omega + (\\alpha + \\gamma I_{t-1}) \\epsilon_{t-1}^2 + \\beta \\sigma_{t-1}^2$$
    
    * **$\\omega$ (Baseline Variance):** Long-run minimum variance constant.
    * **$\\alpha$ (Shock Sensitivity):** Reaction to yesterday's unexpected news/price shock.
    * **$\\gamma$ (Asymmetry / Leverage):** Additional volatility penalty applied when yesterday was a down day ($I_{t-1} = 1$).
    * **$\\beta$ (Persistence):** Memory of past volatility.
    * **Half-Life:** Trading days required for a volatility spike to decay by 50% back to baseline.
    
    **Normal Distribution Probability:**
    * $\\pm 1\\sigma$: ~68.27% of price distribution is contained within this band.
    * $\\pm 2\\sigma$: ~95.45% of price distribution is contained within this band.
    """)

# ---------------------------------------------------------
# HISTORICAL DATA & GARCH ESTIMATION
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

# Multi-Horizon Volatility Scaling (Square-Root of Time Rule)
horizons = {
    "1-Day (Intraday)": daily_sigma * np.sqrt(1),
    "3-Day (Swing)": daily_sigma * np.sqrt(3),
    "7-Day (Weekly Expiry)": daily_sigma * np.sqrt(7),
    "30-Day (Monthly Expiry)": daily_sigma * np.sqrt(30)
}

# ---------------------------------------------------------
# CLEAN CARDS DASHBOARD
# ---------------------------------------------------------
def render_clean_dashboard(current_price: float):
    """Renders clean, chart-free, institutional cards with IST time synchronization."""
    
    # 1. Level Projections
    is_geom = "Geometric" in calc_method
    if is_geom:
        p_plus_2s = current_price * np.exp(2 * daily_sigma)
        p_plus_1s = current_price * np.exp(1 * daily_sigma)
        p_minus_1s = current_price * np.exp(-1 * daily_sigma)
        p_minus_2s = current_price * np.exp(-2 * daily_sigma)
        p_plus_2_5s = current_price * np.exp(2.5 * daily_sigma)
        p_minus_2_5s = current_price * np.exp(-2.5 * daily_sigma)
    else:
        p_plus_2s = current_price * (1 + 2 * daily_sigma)
        p_plus_1s = current_price * (1 + 1 * daily_sigma)
        p_minus_1s = current_price * (1 - 1 * daily_sigma)
        p_minus_2s = current_price * (1 - 2 * daily_sigma)
        p_plus_2_5s = current_price * (1 + 2.5 * daily_sigma)
        p_minus_2_5s = current_price * (1 - 2.5 * daily_sigma)

    tick = SUPPORTED_SYMBOLS[symbol]["tick_size"]
    p_plus_2s_rnd = round(p_plus_2s / tick) * tick
    p_plus_1s_rnd = round(p_plus_1s / tick) * tick
    p_minus_1s_rnd = round(p_minus_1s / tick) * tick
    p_minus_2s_rnd = round(p_minus_2s / tick) * tick

    # VaR 95% and 99%
    var_95_pct = 1.645 * daily_sigma
    var_99_pct = 2.326 * daily_sigma
    var_95_usd = current_price * var_95_pct

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

    # IST Time formatting
    ist_now = datetime.now(IST)
    ist_time_str = ist_now.strftime("%H:%M:%S IST")

    # Header Row
    h_col1, h_col2 = st.columns([2.6, 1.4])
    with h_col1:
        st.title(f"⚡ {symbol} — Quantitative Volatility Terminal")
        pers = garch_diag.get('persistence', 0.0)
        hl = garch_diag.get('half_life', np.nan)
        hl_str = f"{hl:.1f} days" if not np.isnan(hl) else "N/A"
        st.caption(
            f"**Contract:** {SUPPORTED_SYMBOLS[symbol]['name']} | "
            f"**Model:** {garch_diag['method']} | "
            f"**Persistence (α+β):** `{pers:.3f}` | "
            f"**Half-Life:** `{hl_str}`"
        )
    with h_col2:
        st.metric(
            label="Live Delta Reference Price",
            value=f"${current_price:,.2f}",
            delta=f"Live Feed @ {ist_time_str}",
            delta_color="off"
        )

    st.markdown("---")

    # 2. Risk Metrics & Volatility Thermometer
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric(
            "Daily Volatility (1σ)",
            f"{daily_sigma * 100:.3f}%",
            delta=f"Annualized: {annual_sigma * 100:.2f}%",
            delta_color="off",
            help="Forecast 1-day standard deviation from GARCH model."
        )
    with m2:
        st.metric(
            "1-Day 95% VaR (Risk)",
            f"${var_95_usd:,.2f}",
            delta=f"-{var_95_pct * 100:.2f}% Max 1D Loss",
            delta_color="inverse",
            help="Value at Risk: 95% of trading days will experience losses less than this amount."
        )
    with m3:
        st.metric(
            "1-Day 99% Tail Risk",
            f"-{var_99_pct * 100:.2f}%",
            delta=f"-${(current_price * var_99_pct):,.2f}",
            delta_color="inverse",
            help="Stress-test extreme quantile (2.33σ)."
        )
    with m4:
        st.metric(
            "Volatility Regime",
            regime_badge.split(" ")[1],
            delta=regime_badge.split(" ")[0] + " " + " ".join(regime_badge.split(" ")[2:]),
            delta_color="off",
            help="Current forecast annual volatility vs historical long-run baseline."
        )

    st.markdown("---")

    # 3. GARCH Level Cards
    st.markdown("### 🎯 GARCH Volatility Level Cards")

    # +2 Sigma Card
    st.markdown(f"""
    <div class="level-card level-p2">
        <div>
            <div class="level-name" style="color: #ef5350;">🔴 +2 Sigma (+2σ) Resistance</div>
            <div class="level-desc">95.4% Statistical Envelope • Extreme Overbought • Optimal for Selling OTM Calls / Bear Call Spreads</div>
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
            <div class="level-name" style="color: #ffa726;">🟠 +1 Sigma (+1σ) Resistance</div>
            <div class="level-desc">68.3% Statistical Envelope • Momentum Exhaustion Barrier • Take-Profit Zone for Longs</div>
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
            <div class="level-name" style="color: #29b6f6;">🔵 🎯 Base Anchor Reference (Current Mark)</div>
            <div class="level-desc">Market Equilibrium Point • Delta Exchange Live Execution Anchor</div>
        </div>
        <div>
            <div class="level-price" style="color: #29b6f6;">${current_price:,.2f}</div>
            <div class="level-spread" style="color: #29b6f6;">0.00% Center Point</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # -1 Sigma Card
    st.markdown(f"""
    <div class="level-card level-m1">
        <div>
            <div class="level-name" style="color: #66bb6a;">🟢 -1 Sigma (-1σ) Support</div>
            <div class="level-desc">68.3% Statistical Envelope • Trend Pullback Support • Dip Buying Accumulation Zone</div>
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
            <div class="level-name" style="color: #26a69a;">🟢 -2 Sigma (-2σ) Support</div>
            <div class="level-desc">95.4% Statistical Envelope • Extreme Oversold • High Probability Reversal Long / Bull Put Spreads</div>
        </div>
        <div>
            <div class="level-price" style="color: #26a69a;">${p_minus_2s_rnd:,.2f}</div>
            <div class="level-spread" style="color: #26a69a;">{((p_minus_2s/current_price)-1)*100:.2f}% from anchor</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # 4. Multi-Horizon Expected Moves Row
    st.markdown("### ⏳ Multi-Horizon Volatility Term Structure ($\sigma \cdot \sqrt{T}$)")
    
    h_cols = st.columns(4)
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
                help=f"Projected ±2σ (95.4%) expected price movement over {h_name}."
            )

    st.markdown("---")

    # 5. Delta Exchange Order Bracket Generator
    st.markdown("### 🛡️ Delta Exchange Order Bracket Generator")
    st.caption("Quantitative parameters matching Delta Exchange India Bracket Order API structure.")
    
    b_col1, b_col2 = st.columns(2)
    with b_col1:
        st.markdown("**🟢 Bullish Mean-Reversion Plan (Long at -1σ / -2σ)**")
        entry_long = p_minus_1s_rnd
        tp_long = current_price
        sl_long = round(p_minus_2_5s / tick) * tick
        risk_long = entry_long - sl_long
        reward_long = tp_long - entry_long
        rr_long = reward_long / risk_long if risk_long > 0 else 1.0
        
        st.markdown(f"""
        <div class="bracket-box">
            <b>Target Entry:</b> <code>${entry_long:,.2f}</code> (-1σ Support)<br>
            <b>Take-Profit Price:</b> <code>${tp_long:,.2f}</code> (Anchor Baseline)<br>
            <b>Stop-Loss Trigger:</b> <code>${sl_long:,.2f}</code> (-2.5σ Tail Defense)<br>
            <b>Risk/Reward Ratio:</b> <span style="color:#66bb6a; font-weight:700;">1 : {rr_long:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

    with b_col2:
        st.markdown("**🔴 Bearish Mean-Reversion Plan (Short at +1σ / +2σ)**")
        entry_short = p_plus_1s_rnd
        tp_short = current_price
        sl_short = round(p_plus_2_5s / tick) * tick
        risk_short = sl_short - entry_short
        reward_short = entry_short - tp_short
        rr_short = reward_short / risk_short if risk_short > 0 else 1.0
        
        st.markdown(f"""
        <div class="bracket-box">
            <b>Target Entry:</b> <code>${entry_short:,.2f}</code> (+1σ Resistance)<br>
            <b>Take-Profit Price:</b> <code>${tp_short:,.2f}</code> (Anchor Baseline)<br>
            <b>Stop-Loss Trigger:</b> <code>${sl_short:,.2f}</code> (+2.5σ Tail Defense)<br>
            <b>Risk/Reward Ratio:</b> <span style="color:#ef5350; font-weight:700;">1 : {rr_short:.2f}</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # 6. Quantitative Execution Matrix Table
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
        "Normal Distribution Prob": [
            "~2.3% Tail Prob (Overbought)",
            "~15.9% Upper Boundary",
            "50.0% Median Fair Value",
            "~15.9% Lower Boundary",
            "~2.3% Tail Prob (Oversold)"
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
# AUTO-REFRESH EXECUTION CONTROLLER
# ---------------------------------------------------------
if auto_refresh and hasattr(st, "fragment"):
    @st.fragment(run_every=f"{refresh_rate_sec}s")
    def live_auto_refresh_view():
        if use_manual_override:
            curr_px = float(st.session_state[session_px_key])
        else:
            fresh_px, _ = fetch_live_ticker(symbol)
            curr_px = fresh_px if (fresh_px and fresh_px > 0) else base_price
        render_clean_dashboard(curr_px)
        
    live_auto_refresh_view()

elif auto_refresh:
    # Fallback for older Streamlit environments
    if use_manual_override:
        curr_px = float(st.session_state[session_px_key])
    else:
        fresh_px, _ = fetch_live_ticker(symbol)
        curr_px = fresh_px if (fresh_px and fresh_px > 0) else base_price
    render_clean_dashboard(curr_px)
    time.sleep(refresh_rate_sec)
    st.rerun()

else:
    # Manual static view
    render_clean_dashboard(base_price)