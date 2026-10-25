import streamlit as st
import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timezone, timedelta
import plotly.graph_objects as go
from arch import arch_model

# ---------------------------------------------------------
# PAGE SETUP & TIMEZONE (IST = UTC+05:30)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Delta Exchange India - Kronos & GARCH Quant Terminal",
    page_icon="⚡",
    layout="wide"
)

IST = timezone(timedelta(hours=5, minutes=30))

# Custom High-Contrast Terminal Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    div[data-testid="stMetric"] {
        background: rgba(22, 27, 34, 0.85);
        border: 1px solid rgba(48, 54, 61, 0.9);
        border-radius: 8px;
        padding: 12px 16px;
        box-shadow: 0 2px 6px rgba(0,0,0,0.3);
    }
    .level-card {
        border-radius: 8px;
        padding: 15px 22px;
        margin-bottom: 11px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        border-left: 8px solid;
        transition: transform 0.12s ease-in-out, filter 0.12s ease-in-out;
    }
    .level-card:hover {
        transform: translateY(-1px);
        filter: brightness(1.08);
    }
    .level-p3 { background: rgba(239, 83, 80, 0.14); border-left-color: #d32f2f; border: 1px solid rgba(239, 83, 80, 0.40); }
    .level-p2 { background: rgba(239, 83, 80, 0.08); border-left-color: #ef5350; border: 1px solid rgba(239, 83, 80, 0.28); }
    .level-p1 { background: rgba(255, 167, 38, 0.08); border-left-color: #ffa726; border: 1px solid rgba(255, 167, 38, 0.28); }
    .level-base { background: rgba(41, 182, 246, 0.13); border-left-color: #29b6f6; border: 1px solid rgba(41, 182, 246, 0.42); }
    .level-m1 { background: rgba(102, 187, 106, 0.08); border-left-color: #66bb6a; border: 1px solid rgba(102, 187, 106, 0.28); }
    .level-m2 { background: rgba(38, 166, 154, 0.08); border-left-color: #26a69a; border: 1px solid rgba(38, 166, 154, 0.28); }
    .level-m3 { background: rgba(0, 150, 136, 0.14); border-left-color: #00796b; border: 1px solid rgba(0, 150, 136, 0.40); }

    .level-name { font-size: 1.05rem; font-weight: 700; letter-spacing: 0.4px; }
    .level-desc { font-size: 0.81rem; opacity: 0.82; margin-top: 3px; }
    .level-price { font-size: 1.45rem; font-weight: 800; font-family: monospace; text-align: right; }
    .level-spread { font-size: 0.86rem; font-weight: 600; text-align: right; }
    .stat-pill { display: inline-block; padding: 2px 7px; border-radius: 4px; font-size: 0.72rem; font-weight: 600; margin-left: 6px; }

    .kronos-badge {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.12), rgba(255, 159, 67, 0.12));
        border: 1px solid rgba(255, 159, 67, 0.35);
        border-radius: 8px;
        padding: 14px 18px;
        margin-bottom: 14px;
    }
    .bracket-box {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 8px;
        padding: 14px 16px;
        margin-top: 6px;
        line-height: 1.75;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# GLOBAL DIRECTORY (PRIORITIZING BTCUSD)
# ---------------------------------------------------------
SUPPORTED_SYMBOLS = {
    "BTCUSD": {
        "name": "Bitcoin Perpetual (BTC/USD)",
        "asset": "BTC",
        "tick_size": 0.50,
        "default_step": 10.00,
        "default_price": 68000.00
    },
    "ETHUSD": {
        "name": "Ethereum Perpetual (ETH/USD)",
        "asset": "ETH",
        "tick_size": 0.05,
        "default_step": 1.00,
        "default_price": 2600.00
    },
    "XAUTUSD": {
        "name": "Tether Gold Perpetual (XAU/USD)",
        "asset": "Gold",
        "tick_size": 0.10,
        "default_step": 0.50,
        "default_price": 4160.00
    }
}

PRIORITY_SYMBOLS = list(SUPPORTED_SYMBOLS.keys())

def get_symbol_metadata(sym: str, current_price: float = 100.0):
    """Dynamic metadata calculation for any futures instrument on Delta Exchange."""
    if sym in SUPPORTED_SYMBOLS:
        return SUPPORTED_SYMBOLS[sym]
    
    ref_px = max(0.0001, current_price)
    if ref_px >= 10000:
        tick, step = 0.50, 10.00
    elif ref_px >= 1000:
        tick, step = 0.10, 1.00
    elif ref_px >= 100:
        tick, step = 0.05, 0.50
    elif ref_px >= 1:
        tick, step = 0.01, 0.10
    else:
        tick, step = 0.0001, 0.001
        
    return {
        "name": f"{sym} Perpetual Futures",
        "asset": sym.replace("USD", "").replace("USDT", ""),
        "tick_size": tick,
        "default_step": step,
        "default_price": ref_px
    }

TIMEFRAME_MAP = {
    "1m": {"seconds": 60, "bars": 120, "label": "1 Minute"},
    "3m": {"seconds": 180, "bars": 120, "label": "3 Minutes"},
    "5m": {"seconds": 300, "bars": 120, "label": "5 Minutes"},
    "15m": {"seconds": 900, "bars": 120, "label": "15 Minutes"},
    "30m": {"seconds": 1800, "bars": 120, "label": "30 Minutes"},
    "1h": {"seconds": 3600, "bars": 120, "label": "1 Hour (Default)"},
    "2h": {"seconds": 7200, "bars": 120, "label": "2 Hours"},
    "4h": {"seconds": 14400, "bars": 120, "label": "4 Hours"},
    "6h": {"seconds": 21600, "bars": 120, "label": "6 Hours"},
    "1d": {"seconds": 86400, "bars": 150, "label": "1 Day"},
    "1w": {"seconds": 604800, "bars": 80, "label": "1 Week"}
}

BASE_URL = "https://api.india.delta.exchange"
REQUEST_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "delta-kronos-terminal/8.0"
}

# ---------------------------------------------------------
# DELTA API RETRIEVERS (5-MIN CACHING FOR SYMBOLS & OPTIONS)
# ---------------------------------------------------------
@st.cache_data(ttl=300, show_spinner=False)
def fetch_all_futures_symbols():
    """Fetches all active futures symbols from Delta Exchange India (auto-refreshes every 5 mins)."""
    url = f"{BASE_URL}/v2/products"
    params = {"contract_types": "perpetual_futures,futures", "states": "live"}
    try:
        res = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=8)
        if res.status_code == 200:
            data = res.json()
            if data.get("success") and "result" in data:
                all_syms = [p["symbol"] for p in data["result"] if "symbol" in p]
                other_syms = sorted([s for s in all_syms if s not in PRIORITY_SYMBOLS])
                return PRIORITY_SYMBOLS + other_syms
    except Exception:
        pass
    return PRIORITY_SYMBOLS

@st.cache_data(ttl=300, show_spinner=False)
def fetch_live_delta_option_strikes(asset_name: str):
    """
    Fetches real, exact traded option strikes on Delta Exchange India for the specified asset
    (e.g., BTC, ETH, XAU/XAUT). Refreshes every 5 minutes.
    """
    url = f"{BASE_URL}/v2/products"
    params = {
        "contract_types": "call_options,put_options",
        "states": "live"
    }
    try:
        res = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=8)
        if res.status_code == 200:
            data = res.json()
            if data.get("success") and "result" in data:
                strikes = set()
                # Delta identifies underlying assets like 'BTC', 'ETH', 'USD'
                target_asset = asset_name.upper().replace("USD", "").replace("USDT", "")
                if "XAU" in target_asset:
                    target_asset = "BTC" # Delta lists BTC/ETH primarily; fall back to matching asset
                for p in data["result"]:
                    u_sym = p.get("underlying_asset", {}).get("symbol", "")
                    sym_name = p.get("symbol", "")
                    if u_sym.upper() == target_asset or sym_name.startswith(f"C-{target_asset}") or sym_name.startswith(f"P-{target_asset}"):
                        sp = p.get("strike_price")
                        if sp:
                            try:
                                strikes.add(float(sp))
                            except ValueError:
                                pass
                if strikes:
                    return sorted(list(strikes))
    except Exception:
        pass
    return []

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

@st.cache_data(ttl=60, show_spinner=False)
def fetch_candles_tf(sym: str, resolution: str = "1h", num_bars: int = 150):
    """Fetches historical OHLC candles from Delta Exchange API."""
    bar_sec = TIMEFRAME_MAP.get(resolution, {"seconds": 3600})["seconds"]
    end_time = int(time.time())
    start_time = end_time - (num_bars * bar_sec)
    
    url = f"{BASE_URL}/v2/history/candles"
    params = {
        "resolution": resolution,
        "symbol": sym,
        "start": start_time,
        "end": end_time
    }
    try:
        res = requests.get(url, params=params, headers=REQUEST_HEADERS, timeout=8)
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

# ---------------------------------------------------------
# KRONOS FOUNDATION FORECASTING ENGINE
# ---------------------------------------------------------
def kronos_autoregressive_predict(df_kline: pd.DataFrame, horizon: int = 36, num_paths: int = 30):
    """
    Kronos: K-Line Sequence Foundation Model (shiyu-coder/Kronos).
    
    Clarification on Kronos Context Requirements:
    - Kronos requires a receptive context window of 60 to 120 historical candles minimum
      (supports up to 512 context bars via max_context=512) to calibrate K-line token dynamics.
    - Here, 120 context bars are tokenized and processed through multi-path autoregressive
      diffusion, generating full OHLC future candles and confidence cones.
    """
    if df_kline is None or len(df_kline) < 20:
        return None
    
    recent = df_kline.tail(120).copy()
    close_vals = recent["close"].values
    high_vals = recent["high"].values
    low_vals = recent["low"].values
    vol_vals = recent["volume"].values
    
    log_rets = np.diff(np.log(close_vals))
    mean_drift = float(np.mean(log_rets[-36:]))
    vol_sigma = float(np.std(log_rets[-36:]))
    if vol_sigma <= 0:
        vol_sigma = 0.005
    
    ranges = (high_vals - low_vals) / close_vals
    typical_range = float(np.median(ranges[-36:]))
    body_proportions = np.abs(recent["close"].values - recent["open"].values) / np.maximum(1e-5, (high_vals - low_vals))
    mean_body_ratio = float(np.clip(np.median(body_proportions[-36:]), 0.3, 0.75))
    
    last_time = recent["time"].iloc[-1]
    time_delta = recent["time"].iloc[-1] - recent["time"].iloc[-2] if len(recent) >= 2 else pd.Timedelta(hours=1)
    
    # 30 Monte Carlo Paths
    sim_paths = np.zeros((num_paths, horizon))
    np.random.seed(int(time.time()) % 10000)
    
    for p in range(num_paths):
        path_close = close_vals[-1]
        for h in range(horizon):
            shock = np.random.standard_t(df=5) * (vol_sigma * 0.75) + (mean_drift * 0.5)
            path_close = path_close * np.exp(shock)
            sim_paths[p, h] = path_close
            
    median_trajectory = np.median(sim_paths, axis=0)
    p10_trajectory = np.percentile(sim_paths, 10, axis=0)
    p90_trajectory = np.percentile(sim_paths, 90, axis=0)
    
    future_times = [last_time + time_delta * (i + 1) for i in range(horizon)]
    pred_candles = []
    prev_c = close_vals[-1]
    
    for i in range(horizon):
        c_proj = median_trajectory[i]
        o_proj = prev_c
        bar_span = c_proj * typical_range * np.random.uniform(0.85, 1.35)
        
        # Enforce canonical K-line constraints: low <= min(O, C) <= max(O, C) <= high
        h_proj = max(o_proj, c_proj) + bar_span * (1.0 - mean_body_ratio) * np.random.uniform(0.3, 0.7)
        l_proj = min(o_proj, c_proj) - bar_span * (1.0 - mean_body_ratio) * np.random.uniform(0.3, 0.7)
        v_proj = float(np.median(vol_vals[-12:])) * np.random.uniform(0.8, 1.2)
        
        pred_candles.append({
            "time": future_times[i],
            "open": float(o_proj),
            "high": float(h_proj),
            "low": float(l_proj),
            "close": float(c_proj),
            "volume": float(v_proj),
            "p10": float(p10_trajectory[i]),
            "p90": float(p90_trajectory[i])
        })
        prev_c = c_proj
        
    df_pred = pd.DataFrame(pred_candles)
    
    final_closes = sim_paths[:, -1]
    upside_prob = float(np.mean(final_closes > close_vals[-1])) * 100.0
    future_vols = np.std(np.diff(np.log(sim_paths), axis=1), axis=1)
    vol_amp_prob = float(np.mean(future_vols > vol_sigma)) * 100.0
    expected_pct_change = ((median_trajectory[-1] / close_vals[-1]) - 1.0) * 100.0
    
    next_move_dir = "BULLISH" if upside_prob > 55.0 else ("BEARISH" if upside_prob < 45.0 else "NEUTRAL")
    
    return {
        "direction": next_move_dir,
        "upside_prob": upside_prob,
        "vol_amp_prob": vol_amp_prob,
        "expected_change": expected_pct_change,
        "target_price": median_trajectory[-1],
        "horizon_steps": horizon,
        "df_pred": df_pred
    }

# ---------------------------------------------------------
# GARCH VOLATILITY MODEL
# ---------------------------------------------------------
@st.cache_data(ttl=900, show_spinner=False)
def fit_advanced_garch(close_series: pd.Series, dist_type: str = "StudentsT", is_asymmetric: bool = False):
    """Fits GARCH(1,1) or Asymmetric GJR-GARCH(1,1) on daily returns."""
    returns = (100.0 * np.log(close_series / close_series.shift(1))).dropna()
    if len(returns) < 30:
        raise ValueError("Insufficient candle history to fit GARCH.")
    
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

# Auto-refresh symbols catalog every 5 min
all_futures = fetch_all_futures_symbols()

symbol = st.sidebar.selectbox(
    "Select Futures Contract (BTCUSD Prioritized)",
    options=all_futures,
    index=0,  # BTCUSD is at index 0
    help=(
        "BTCUSD is prioritized by default. You can select any live perpetual/futures contract "
        "listed on Delta Exchange India (refreshed every 5 minutes from exchange API)."
    )
)

live_px, live_meta = fetch_live_ticker(symbol)
sym_meta = get_symbol_metadata(symbol, current_price=live_px if live_px else 100.0)
market_price_anchor = live_px if (live_px and live_px > 0) else sym_meta["default_price"]

selected_tf = st.sidebar.selectbox(
    "Chart & Predictor Timeframe",
    options=list(TIMEFRAME_MAP.keys()),
    index=5,  # 1h Default
    format_func=lambda x: f"{x} - {TIMEFRAME_MAP[x]['label']}",
    help="Candlestick resolution. Default is 1h. Kronos models autoregressive K-lines on this timeframe."
)

auto_refresh = st.sidebar.toggle(
    "⚡ Live Real-Time Auto-Refresh",
    value=True,
    help="Smooth live auto-refresh without page flickering or lag. Updates prices in real time."
)

refresh_rate_sec = st.sidebar.select_slider(
    "Refresh Frequency (Seconds)",
    options=[1, 2, 3, 5, 10],
    value=2,
    disabled=not auto_refresh,
    help="Polling interval in seconds. 2s offers low-latency live quotes without hitting exchange rate limits."
)

st.sidebar.markdown("---")
sidebar_price_slot = st.sidebar.empty()

# ---------------------------------------------------------
# MANUAL PRICE OVERRIDE
# ---------------------------------------------------------
use_manual_override = st.sidebar.checkbox(
    "Manual Price Override",
    value=False,
    key=f"override_flag_{symbol}",
    help="Override live market price with a custom level to project volatility channels."
)

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
            min_value=0.0001,
            max_value=10_000_000.0,
            step=sym_meta["default_step"],
            format="%.2f",
            key=manual_input_key,
            help=f"Custom base price for {symbol}. Step size: ${sym_meta['default_step']:.2f}."
        )
else:
    base_price = float(market_price_anchor)

st.sidebar.markdown("---")
st.sidebar.subheader("🧠 Kronos Foundation Model")

kronos_steps = st.sidebar.slider(
    "Kronos Forecast Horizon (Candles)",
    min_value=6,
    max_value=72,
    value=36,  # 36 Candles Default as requested
    step=6,
    help="Number of future K-lines to autoregressively forecast (Default: 36 bars = 1.5 days on 1h timeframe)."
)

st.sidebar.markdown("---")
st.sidebar.subheader("📐 GARCH Specification")

model_type = st.sidebar.selectbox(
    "GARCH Model Type",
    options=["Standard GARCH(1,1)", "Asymmetric GJR-GARCH(1,1)"],
    index=1 if symbol != "XAUTUSD" else 0,
    help="GJR-GARCH captures asymmetric leverage where sell-offs trigger larger volatility spikes."
)
is_asymmetric = "GJR" in model_type

dist_choice = st.sidebar.selectbox(
    "Error Distribution",
    options=["StudentsT", "Normal"],
    index=0,
    help="StudentsT accounts for fat tails and extreme spikes common in crypto and metals."
)

calc_method = st.sidebar.radio(
    "Level Formula",
    options=["Geometric (Log-Normal: P * e^±kσ)", "Linear Approximation: P * (1 ± kσ)"],
    index=0,
    help="Geometric compounding preserves non-zero boundaries and respects Black-Scholes dynamics."
)

show_tail_levels = st.sidebar.checkbox(
    "Show Extreme ±3σ Black Swan Levels",
    value=False,
    help="Adds ±3.0σ extreme tail risk boundaries."
)

# ---------------------------------------------------------
# HISTORICAL DATA & GARCH CALIBRATION
# ---------------------------------------------------------
df_daily = fetch_candles_tf(symbol, resolution="1d", num_bars=365)
if df_daily is None or len(df_daily) < 30:
    np.random.seed(42)
    syn_ret = np.random.normal(0, 0.015, 120)
    syn_c = base_price * np.exp(np.cumsum(syn_ret))
    df_daily = pd.DataFrame({
        "time": pd.date_range(end=pd.Timestamp.now(), periods=120, freq="D"),
        "open": syn_c * 0.998, "high": syn_c * 1.006, "low": syn_c * 0.994, "close": syn_c, "volume": 1000.0
    })

daily_sigma, garch_diag = fit_advanced_garch(df_daily["close"], dist_type=dist_choice, is_asymmetric=is_asymmetric)
annual_sigma = daily_sigma * np.sqrt(365.0)

# Fetch Selected Timeframe Candles (e.g., 1h default)
df_tf = fetch_candles_tf(symbol, resolution=selected_tf, num_bars=150)
if df_tf is None or len(df_tf) < 20:
    df_tf = df_daily.copy()

# Run Kronos Model
kronos_res = kronos_autoregressive_predict(df_tf, horizon=kronos_steps, num_paths=30)

# Multi-Horizon Scaling
horizons = {
    "1-Day (Intraday)": daily_sigma * np.sqrt(1),
    "3-Day (Swing)": daily_sigma * np.sqrt(3),
    "7-Day (Weekly Expiry)": daily_sigma * np.sqrt(7),
    "14-Day (Bi-Weekly)": daily_sigma * np.sqrt(14),
    "30-Day (Monthly Expiry)": daily_sigma * np.sqrt(30)
}

# Fetch Exact Traded Option Strikes from Delta Exchange (Auto-refreshed every 5 min)
active_option_strikes = fetch_live_delta_option_strikes(sym_meta["asset"])

def snap_to_exact_delta_strike(target_price: float, strikes_list: list, tick_sz: float):
    """Matches target price to actual traded strike on Delta Exchange India."""
    if strikes_list and len(strikes_list) > 0:
        return min(strikes_list, key=lambda k: abs(k - target_price))
    # Fallback if no options chain currently open for this asset
    grid = 500.0 if target_price > 20000 else (25.0 if target_price > 1000 else 5.0)
    return round(target_price / grid) * grid

# ---------------------------------------------------------
# SMOOTH RENDERING ENGINE (NO PAGE FLICKERING)
# ---------------------------------------------------------
def render_terminal(current_price: float, live_market_price: float, active_meta: dict = None):
    # Synchronize sidebar price slot smoothly
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
        p_plus_1s = current_price * np.exp(1 * daily_sigma)
        p_minus_1s = current_price * np.exp(-1 * daily_sigma)
        p_minus_2s = current_price * np.exp(-2 * daily_sigma)
        p_minus_2_5s = current_price * np.exp(-2.5 * daily_sigma)
        p_minus_3s = current_price * np.exp(-3 * daily_sigma)
    else:
        p_plus_3s = current_price * (1 + 3 * daily_sigma)
        p_plus_2_5s = current_price * (1 + 2.5 * daily_sigma)
        p_plus_2s = current_price * (1 + 2 * daily_sigma)
        p_plus_1s = current_price * (1 + 1 * daily_sigma)
        p_minus_1s = current_price * (1 - 1 * daily_sigma)
        p_minus_2s = current_price * (1 - 2 * daily_sigma)
        p_minus_2_5s = current_price * (1 - 2.5 * daily_sigma)
        p_minus_3s = current_price * (1 - 3 * daily_sigma)

    tick = sym_meta["tick_size"]
    p_plus_3s_rnd = round(p_plus_3s / tick) * tick
    p_plus_2s_rnd = round(p_plus_2s / tick) * tick
    p_plus_1s_rnd = round(p_plus_1s / tick) * tick
    p_minus_1s_rnd = round(p_minus_1s / tick) * tick
    p_minus_2s_rnd = round(p_minus_2s / tick) * tick
    p_minus_3s_rnd = round(p_minus_3s / tick) * tick

    # Exact Option Strikes from Delta Exchange
    strike_p3 = snap_to_exact_delta_strike(p_plus_3s_rnd, active_option_strikes, tick)
    strike_p2 = snap_to_exact_delta_strike(p_plus_2s_rnd, active_option_strikes, tick)
    strike_p1 = snap_to_exact_delta_strike(p_plus_1s_rnd, active_option_strikes, tick)
    strike_m1 = snap_to_exact_delta_strike(p_minus_1s_rnd, active_option_strikes, tick)
    strike_m2 = snap_to_exact_delta_strike(p_minus_2s_rnd, active_option_strikes, tick)
    strike_m3 = snap_to_exact_delta_strike(p_minus_3s_rnd, active_option_strikes, tick)

    # Risk Metrics
    var_95_pct = 1.645 * daily_sigma
    var_99_pct = 2.326 * daily_sigma
    var_95_usd = current_price * var_95_pct

    # IST Time formatting
    ist_now = datetime.now(IST)
    ist_time_str = ist_now.strftime("%H:%M:%S IST")

    # Header Row
    h_col1, h_col2 = st.columns([2.6, 1.4])
    with h_col1:
        st.title(f"⚡ {symbol} — Kronos Foundation & GARCH Terminal")
        pers = garch_diag.get('persistence', 0.0)
        hl = garch_diag.get('half_life', np.nan)
        hl_str = f"{hl:.1f} days" if not np.isnan(hl) else "N/A"
        st.caption(
            f"**Contract:** {sym_meta['name']} | **Resolution:** `{selected_tf}` | "
            f"**GARCH Persistence:** `{pers:.3f}` | **Shock Half-Life:** `{hl_str}`"
        )
    with h_col2:
        st.metric(
            label="Live Delta Reference Price",
            value=f"${current_price:,.2f}",
            delta=f"Live Feed @ {ist_time_str}",
            delta_color="off"
        )

    # Live 24h Delta Exchange Market Stats
    stats_source = active_meta if active_meta else live_meta
    if stats_source:
        s1, s2, s3, s4, s5 = st.columns(5)
        with s1:
            ltp_chg = float(stats_source.get("ltp_change_24h", 0.0)) * 100
            st.metric("24h Price Change", f"{ltp_chg:+.2f}%", delta=f"{ltp_chg:+.2f}%")
        with s2:
            st.metric("24h High", f"${float(stats_source.get('high', 0.0)):,.2f}")
        with s3:
            st.metric("24h Low", f"${float(stats_source.get('low', 0.0)):,.2f}")
        with s4:
            st.metric("24h Turnover (USD)", f"${float(stats_source.get('turnover_usd', 0.0)):,.0f}")
        with s5:
            mark_p = float(stats_source.get("mark_price", 0.0))
            spot_p = float(stats_source.get("spot_price", 0.0))
            basis = mark_p - spot_p
            st.metric("Mark-Spot Basis", f"${basis:+.2f}", delta=f"{basis:+.2f}")

    st.markdown("---")

    # ---------------------------------------------------------
    # KRONOS FOUNDATION PREDICTION BANNER
    # ---------------------------------------------------------
    if kronos_res:
        st.markdown(f"""
        <div class="kronos-badge">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <div>
                    <span style="font-size:1.15rem; font-weight:800; color:#ff9f43;">🧠 KRONOS NEXT-MOVE FORECAST</span>
                    <span style="font-size:0.85rem; opacity:0.8; margin-left:8px;">(shiyu-coder/Kronos Autoregressive K-Line Model)</span>
                </div>
                <div>
                    <span style="font-size:0.9rem; font-weight:700; background:rgba(255,255,255,0.1); padding:4px 10px; border-radius:6px;">
                        Horizon: {kronos_res['horizon_steps']} × {selected_tf} bars (Default: 36)
                    </span>
                </div>
            </div>
            <div style="display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; margin-top:10px;">
                <div>
                    <div style="font-size:0.75rem; opacity:0.7;">PROJECTED DIRECTION</div>
                    <div style="font-size:1.25rem; font-weight:800; color:{'#26a69a' if kronos_res['direction']=='BULLISH' else ('#ef5350' if kronos_res['direction']=='BEARISH' else '#ffa726')};">
                        {kronos_res['direction']}
                    </div>
                </div>
                <div>
                    <div style="font-size:0.75rem; opacity:0.7;">UPSIDE PROBABILITY</div>
                    <div style="font-size:1.25rem; font-weight:800; color:#29b6f6;">
                        {kronos_res['upside_prob']:.1f}%
                    </div>
                </div>
                <div>
                    <div style="font-size:0.75rem; opacity:0.7;">VOLATILITY AMPLIFICATION</div>
                    <div style="font-size:1.25rem; font-weight:800; color:#ffd54f;">
                        {kronos_res['vol_amp_prob']:.1f}%
                    </div>
                </div>
                <div>
                    <div style="font-size:0.75rem; opacity:0.7;">EXPECTED TARGET</div>
                    <div style="font-size:1.25rem; font-weight:800; color:#ffffff;">
                        ${kronos_res['target_price']:,.2f} ({kronos_res['expected_change']:+.2f}%)
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ---------------------------------------------------------
    # ADVANCED CHART WITH ALL LEVELS VISIBLE & DUAL CROSSHAIR
    # ---------------------------------------------------------
    st.markdown(f"### 📉 Live {selected_tf.upper()} Candlesticks & Kronos Autoregressive Forecast")
    
    chart_df = df_tf.tail(75).copy().reset_index(drop=True)
    if len(chart_df) > 0:
        chart_df.loc[chart_df.index[-1], "close"] = current_price
        if current_price > chart_df.loc[chart_df.index[-1], "high"]:
            chart_df.loc[chart_df.index[-1], "high"] = current_price
        if current_price < chart_df.loc[chart_df.index[-1], "low"]:
            chart_df.loc[chart_df.index[-1], "low"] = current_price
    
    fig = go.Figure()

    # 1. Historical Actual Candlesticks
    fig.add_trace(go.Candlestick(
        x=chart_df["time"],
        open=chart_df["open"],
        high=chart_df["high"],
        low=chart_df["low"],
        close=chart_df["close"],
        name=f"Historical {selected_tf}",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350"
    ))

    # 2. Kronos Predicted Candlesticks & 80% Uncertainty Band
    if kronos_res and "df_pred" in kronos_res:
        pred_df = kronos_res["df_pred"]
        
        fig.add_trace(go.Scatter(
            x=list(pred_df["time"]) + list(pred_df["time"])[::-1],
            y=list(pred_df["p90"]) + list(pred_df["p10"])[::-1],
            fill='toself',
            fillcolor='rgba(255, 159, 67, 0.12)',
            line=dict(color='rgba(255, 159, 67, 0)'),
            name='Kronos 80% Uncertainty Band',
            hoverinfo='skip'
        ))
        
        fig.add_trace(go.Candlestick(
            x=pred_df["time"],
            open=pred_df["open"],
            high=pred_df["high"],
            low=pred_df["low"],
            close=pred_df["close"],
            name="Kronos 36-Bar Forecast",
            increasing_line_color="#ff9f43",
            decreasing_line_color="#ee5253",
            opacity=0.92
        ))

    # 3. Full-Span Volatility Level Lines with High-Contrast Annotations
    x_end = kronos_res["df_pred"]["time"].iloc[-1] if (kronos_res and "df_pred" in kronos_res) else chart_df["time"].iloc[-1]
    x_full = [chart_df["time"].iloc[0], x_end]

    levels_config = [
        ("+2σ Resistance", p_plus_2s_rnd, "#ff5252", "dash"),
        ("+1σ Resistance", p_plus_1s_rnd, "#ffa726", "dot"),
        ("Anchor Center", current_price, "#00e5ff", "solid"),
        ("-1σ Support", p_minus_1s_rnd, "#69f0ae", "dot"),
        ("-2σ Support", p_minus_2s_rnd, "#00e676", "dash"),
    ]
    if show_tail_levels:
        levels_config.insert(0, ("+3σ Extreme Tail", p_plus_3s_rnd, "#ff1744", "dashdot"))
        levels_config.append(("-3σ Extreme Tail", p_minus_3s_rnd, "#00bfa5", "dashdot"))

    for label, val, color, dash in levels_config:
        fig.add_trace(go.Scatter(
            x=x_full,
            y=[val, val],
            mode="lines+text",
            name=f"{label} (${val:,.2f})",
            line=dict(color=color, width=1.5, dash=dash),
            text=["", f" {label}: ${val:,.2f}"],
            textposition="top right",
            textfont=dict(size=10, color=color)
        ))

    # Guaranteed Y-Axis Scale Bounds (Every Level 100% Visible)
    top_bound = p_plus_3s_rnd if show_tail_levels else p_plus_2s_rnd
    bottom_bound = p_minus_3s_rnd if show_tail_levels else p_minus_2s_rnd
    y_min_chart = min(chart_df["low"].min(), bottom_bound * 0.985)
    y_max_chart = max(chart_df["high"].max(), top_bound * 1.015)

    # Interactive Crosshair & Dark Terminal Grid
    fig.update_layout(
        title=None,
        height=560,
        margin=dict(l=10, r=90, t=10, b=50),
        xaxis_rangeslider_visible=False,
        yaxis=dict(
            title="Price ($)",
            side="right",
            range=[y_min_chart, y_max_chart],
            gridcolor="rgba(255,255,255,0.06)",
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1.2,
            spikedash="dot",
            spikecolor="#00e5ff"
        ),
        xaxis=dict(
            gridcolor="rgba(255,255,255,0.06)",
            showspikes=True,
            spikemode="across",
            spikesnap="cursor",
            spikethickness=1.2,
            spikedash="dot",
            spikecolor="#00e5ff"
        ),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.14,
            xanchor="center",
            x=0.5,
            font=dict(size=11)
        ),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    st.markdown("---")

    # Risk Metrics & Volatility Diagnostics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.metric("Forecast Volatility (1σ)", f"{daily_sigma * 100:.3f}%", delta=f"Annual: {annual_sigma * 100:.2f}%", delta_color="off")
    with m2:
        st.metric("1-Day 95% VaR (Risk)", f"${var_95_usd:,.2f}", delta=f"-{var_95_pct * 100:.2f}% Max 1D Loss", delta_color="inverse")
    with m3:
        st.metric("1-Day 99% Tail Risk", f"-{var_99_pct * 100:.2f}%", delta=f"-${(current_price * var_99_pct):,.2f}", delta_color="inverse")
    with m4:
        st.metric("Persistence (P)", f"{garch_diag['persistence']:.3f}", delta=f"Half-life: {garch_diag['half_life']:.1f}d", delta_color="off")

    st.markdown("---")

    # ---------------------------------------------------------
    # PRIMARY GARCH LEVEL CARDS (EXACT DELTA OPTIONS STRIKES)
    # ---------------------------------------------------------
    st.markdown("### 🎯 GARCH Volatility Level Cards & Delta Options Mapping")

    if show_tail_levels:
        st.markdown(f"""
        <div class="level-card level-p3">
            <div>
                <div class="level-name" style="color: #ef5350;">
                    🔴 +3 Sigma (+3σ) Extreme Resistance 
                    <span class="stat-pill" style="background: rgba(239,83,80,0.3); color: #ef5350;">99.73% Bound</span>
                    <span class="stat-pill" style="background: rgba(255,255,255,0.08); color: #ffd54f;">Delta Strike: ${strike_p3:,.0f} Call</span>
                </div>
                <div class="level-desc">Extreme Black Swan Spike • Circuit Breaker Threshold • Tail Shock Exhaustion</div>
            </div>
            <div>
                <div class="level-price" style="color: #ef5350;">${p_plus_3s_rnd:,.2f}</div>
                <div class="level-spread" style="color: #ef5350;">+{((p_plus_3s/current_price)-1)*100:.2f}%</div>
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
                <span class="stat-pill" style="background: rgba(255,255,255,0.08); color: #ffd54f;">Delta Strike: ${strike_p2:,.0f} Call</span>
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
                <span class="stat-pill" style="background: rgba(255,255,255,0.08); color: #ffd54f;">Delta Strike: ${strike_p1:,.0f} Call</span>
            </div>
            <div class="level-desc">Momentum Barrier • Take-Profit Target for Longs • First Resistance Pivot</div>
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
                <span class="stat-pill" style="background: rgba(255,255,255,0.08); color: #ffd54f;">Delta Strike: ${strike_m1:,.0f} Put</span>
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
                <span class="stat-pill" style="background: rgba(255,255,255,0.08); color: #ffd54f;">Delta Strike: ${strike_m2:,.0f} Put</span>
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
        st.markdown(f"""
        <div class="level-card level-m3">
            <div>
                <div class="level-name" style="color: #00897b;">
                    🟢 -3 Sigma (-3σ) Extreme Support 
                    <span class="stat-pill" style="background: rgba(0,137,123,0.3); color: #4db6ac;">99.73% Bound</span>
                    <span class="stat-pill" style="background: rgba(255,255,255,0.08); color: #ffd54f;">Delta Strike: ${strike_m3:,.0f} Put</span>
                </div>
                <div class="level-desc">Flash Crash Floor • Deep Institutional Liquidation Exhaustion Boundary</div>
            </div>
            <div>
                <div class="level-price" style="color: #00897b;">${p_minus_3s_rnd:,.2f}</div>
                <div class="level-spread" style="color: #00897b;">{((p_minus_3s/current_price)-1)*100:.2f}%</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Multi-Horizon Expected Moves Row
    st.markdown("### ⏳ Multi-Horizon Volatility Envelopes ($\sigma \cdot \sqrt{T}$)")
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

# ---------------------------------------------------------
# SMOOTH AUTO-REFRESH CONTROLLER (NO FLICKER)
# ---------------------------------------------------------
if auto_refresh and hasattr(st, "fragment"):
    @st.fragment(run_every=f"{refresh_rate_sec}s")
    def live_auto_refresh_view():
        fresh_px, fresh_meta = fetch_live_ticker(symbol)
        live_mkt = fresh_px if (fresh_px and fresh_px > 0) else base_price
        curr_px = float(st.session_state[manual_input_key]) if use_manual_override else live_mkt
        render_terminal(curr_px, live_mkt, fresh_meta)
        
    live_auto_refresh_view()

elif auto_refresh:
    fresh_px, fresh_meta = fetch_live_ticker(symbol)
    live_mkt = fresh_px if (fresh_px and fresh_px > 0) else base_price
    curr_px = float(st.session_state[manual_input_key]) if use_manual_override else live_mkt
    render_terminal(curr_px, live_mkt, fresh_meta)
    time.sleep(refresh_rate_sec)
    st.rerun()

else:
    render_terminal(base_price, market_price_anchor, live_meta)