"""
Covered Call Candidate Scanner
================================
Screens S&P 500 for covered call candidates using yfinance.

IVP proxy: HV Percentile (30-day realized vol vs. trailing 252-day distribution)
— close enough for candidate screening; true IVP requires paid historical IV data.

Install:
    pip install streamlit yfinance pandas numpy

Run:
    streamlit run cc_scanner.py
"""

import time
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

# ─── Page config ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Covered Call Scanner",
    page_icon="📈",
    layout="wide",
)

st.title("📈 Covered Call Candidate Scanner")
st.caption(
    "Universe: S&P 500 | IV proxy: HV Percentile (30-day HV vs. 252-day range) | "
    "Data: Yahoo Finance (15-min delayed)"
)

# ─── Helpers: data loading ────────────────────────────────────────────────────

@st.cache_data(ttl=86_400, show_spinner=False)
def get_sp500_tickers() -> list[str]:
    """Pull current S&P 500 ticker list from Wikipedia with a browser User-Agent."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    try:
        resp = requests.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        table = pd.read_html(resp.text)[0]
        tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
        return tickers
    except Exception:
        # Fallback: hardcoded S&P 500 core holdings (top ~200 by weight, updated 2025)
        return [
            "AAPL","MSFT","NVDA","AMZN","GOOGL","META","TSLA","GOOG","BRK-B","AVGO",
            "JPM","LLY","UNH","V","XOM","MA","COST","HD","PG","NFLX","JNJ","ABBV",
            "BAC","CRM","WMT","CVX","MRK","ORCL","ACN","AMD","PEP","KO","TMO","ADBE",
            "MCD","ABT","CSCO","GE","LIN","DHR","PM","TXN","CAT","IBM","INTU","AMGN",
            "QCOM","GS","RTX","ISRG","SPGI","BLK","NOW","BKNG","AXP","ELV","T","VRTX",
            "SYK","GILD","MDT","LRCX","ADI","REGN","CI","PGR","MU","DE","PANW",
            "SCHW","TJX","BSX","ZTS","AMAT","CB","AON","SO","ETN","CME","ITW","EOG",
            "PLD","WM","MCO","MSI","NOC","APH","MMM","USB","DUK","COF","EMR","PSA",
            "FCX","HCA","FDX","NSC","SHW","TFC","CL","ECL","WMB","OKE","CSX","TEL",
            "PH","CARR","GWW","WELL","ROP","EW","HLT","CTAS","AIG","AFL","DG","FICO",
            "ALL","NKE","GPC","RSG","PWR","FAST","EXC","AME","KEYS","TROW","MTB",
            "WAB","OTIS","HPQ","VRSK","CTSH","LHX","BDX","PPG","XYL","CDW","BALL",
            "NUE","WAT","GIS","SYY","AVB","SBUX","IDXX","DXCM","RVTY",
            "VICI","IRM","CBOE","WEC","WTW","PTC","BAX","MKC","IR","ATO",
            "RF","FITB","STT","HBAN","CFG","KEY","ZION","MTD","A",
            "IQV","CPRT","CINF","LVS","MGM","CZR","WYNN","RCL","CCL","DAL",
            "UAL","AAL","LUV","JBLU","ALK","ABNB","MAR","H","PVH","RL","TPR","VFC",
            "SNPS","CDNS","MPWR","ENPH","FSLR","CEG","VST","NRG","AES","ETR",
            "PPL","CMS","NI","AEE","LNT","EVRG","PNW","NWE","OGE","SR",
        ]


@st.cache_data(ttl=3_600, show_spinner=False)
def download_history(tickers: tuple, period: str = "1y") -> pd.DataFrame:
    """Download adjusted close + volume for a list of tickers."""
    raw = yf.download(
        list(tickers),
        period=period,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    return raw


# ─── Helpers: technical indicators ───────────────────────────────────────────

def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff().dropna()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi_series = 100 - 100 / (1 + rs)
    return float(rsi_series.iloc[-1]) if not rsi_series.empty else np.nan


def hv_30d(series: pd.Series) -> float:
    """30-day annualised historical volatility (%)."""
    log_ret = np.log(series / series.shift(1)).dropna()
    if len(log_ret) < 30:
        return np.nan
    return float(log_ret.rolling(30).std().iloc[-1] * np.sqrt(252) * 100)


def hv_percentile(series: pd.Series, hv_window: int = 30, lookback: int = 252) -> float:
    """
    Rank current 30-day HV against trailing `lookback` daily HV values.
    Returns a 0–100 percentile (HVP).  Used as a free proxy for IVP.
    Uses all available history if < lookback values exist (e.g. 1-year download).
    """
    log_ret = np.log(series / series.shift(1)).dropna()
    rolling_hv = log_ret.rolling(hv_window).std() * np.sqrt(252) * 100
    rolling_hv = rolling_hv.dropna()
    if len(rolling_hv) < 30:          # need at least 30 data points
        return np.nan
    current = rolling_hv.iloc[-1]
    hist = rolling_hv.iloc[-min(lookback, len(rolling_hv) - 1) - 1 : -1]
    if hist.empty:
        return np.nan
    return round(float((hist < current).mean() * 100), 1)


# ─── Helpers: options ─────────────────────────────────────────────────────────

def best_cc_strike(
    symbol: str,
    stock_price: float,
    dte_min: int,
    dte_max: int,
    otm_lo: float,
    otm_hi: float,
) -> dict | None:
    """
    For a given ticker, find the best OTM call in the target DTE window.
    Returns a dict with strike/bid/IV/OI/ann_yield or None.
    """
    try:
        t = yf.Ticker(symbol)
        expirations = t.options
        if not expirations:
            return None

        today = datetime.today()

        for exp_str in expirations:
            exp_date = datetime.strptime(exp_str, "%Y-%m-%d")
            dte = (exp_date - today).days
            if dte < dte_min:
                continue
            if dte > dte_max:
                break  # expirations are sorted ascending

            calls = t.option_chain(exp_str).calls

            # Strike filter: OTM by otm_lo% to otm_hi%
            lo = stock_price * (1 + otm_lo)
            hi = stock_price * (1 + otm_hi)
            candidates = calls[
                (calls["strike"] >= lo)
                & (calls["strike"] <= hi)
                & (calls["bid"] >= 0.30)      # minimum premium worth trading
                & (calls["openInterest"] >= 100)
            ].copy()

            if candidates.empty:
                continue

            # Pick richest bid
            row = candidates.sort_values("bid", ascending=False).iloc[0]
            ann_yield = (row["bid"] / stock_price) / (dte / 365) * 100

            return {
                "expiration": exp_str,
                "dte": dte,
                "strike": float(row["strike"]),
                "bid": float(row["bid"]),
                "iv_pct": round(float(row["impliedVolatility"]) * 100, 1),
                "open_interest": int(row["openInterest"]),
                "ann_yield_pct": round(ann_yield, 1),
            }

    except Exception:
        pass

    return None


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("⚙️ Scan Parameters")

    st.subheader("Universe")
    universe_size = st.slider(
        "S&P 500 — top N stocks (sorted by index weight)",
        min_value=50, max_value=503, value=200, step=50,
    )

    st.subheader("Stock Filters")
    price_min, price_max = st.slider("Price range ($)", 10, 1000, (20, 300))
    vol_min_k = st.number_input("Min 30-day avg volume (K shares)", value=500, step=100)
    rsi_lo, rsi_hi = st.slider("RSI (14)", 10, 90, (35, 65))
    hvp_min = st.slider("Min HV Percentile (IVP proxy)", 0, 90, 30)
    require_above_ema200 = st.checkbox("Must be above 200-day EMA", value=True)

    st.subheader("Options Filters")
    dte_min, dte_max = st.slider("DTE window", 14, 60, (28, 45))
    otm_lo_pct = st.slider("Min OTM % (call strike above stock)", 1, 10, 3)
    otm_hi_pct = st.slider("Max OTM % (call strike above stock)", 5, 30, 15)

    run_btn = st.button("🔍 Run Scan", type="primary", use_container_width=True)

# ─── Main ─────────────────────────────────────────────────────────────────────

if not run_btn:
    st.markdown("""
    ### How it works

    1. **Downloads 1-year price history** for the S&P 500 subset you choose.
    2. **Pre-screens** each stock by price, volume, EMA 200, and RSI — fast, no options calls needed.
    3. **Calculates HV Percentile** (30-day realized vol ranked against its own trailing 252-day history).
       This is a free proxy for IV Percentile: stocks with HVP > 30% tend to have elevated option premiums.
    4. **Pulls options chains** for passing stocks, finds the best OTM call in your DTE window, and reports the annualised premium yield.

    ### Key columns
    | Column | Meaning |
    |--------|---------|
    | **HV 30d %** | 30-day realized volatility (annualised) |
    | **HV Pctile** | Where today's HV sits vs. past 252 days (IVP proxy) |
    | **Strike** | Recommended call strike |
    | **Bid** | Best bid at that strike |
    | **IV %** | Implied vol reported by Yahoo for that option |
    | **OI** | Open interest |
    | **Ann Yield %** | *(Bid / Stock Price) / (DTE / 365) × 100* — annualised premium if NOT assigned |

    > **Not financial advice.** Always verify liquidity in your broker before trading.
    > Earnings within your DTE window is a dealbreaker — check calendars manually.
    """)
    st.stop()

# ── Step 1: load tickers ──────────────────────────────────────────────────────

prog = st.progress(0, text="Fetching S&P 500 ticker list…")

with st.spinner("Loading tickers…"):
    all_tickers = get_sp500_tickers()

universe = tuple(all_tickers[:universe_size])
prog.progress(5, text=f"Universe: {len(universe)} stocks. Downloading 1-year price history…")

# ── Step 2: download price history ───────────────────────────────────────────

with st.spinner("Downloading price history (cached after first run)…"):
    raw = download_history(universe)

prog.progress(35, text="Calculating indicators…")

# Normalise to flat Close/Volume DataFrames regardless of yfinance MultiIndex version
try:
    if isinstance(raw.columns, pd.MultiIndex):
        level0 = raw.columns.get_level_values(0).unique().tolist()
        level1 = raw.columns.get_level_values(1).unique().tolist()
        if "Close" in level0:
            # Standard yfinance 0.2.x: level0=Price, level1=Ticker
            close_df = raw["Close"].copy()
            vol_df   = raw["Volume"].copy() if "Volume" in level0 else pd.DataFrame()
        elif "Close" in level1:
            # Transposed: level0=Ticker, level1=Price
            close_df = raw.xs("Close", level=1, axis=1).copy()
            vol_df   = raw.xs("Volume", level=1, axis=1).copy() if "Volume" in level1 else pd.DataFrame()
        else:
            st.error(f"Unexpected column structure. Level 0: {level0[:5]}, Level 1: {level1[:5]}")
            st.stop()
    else:
        close_df = raw[["Close"]].copy()
        vol_df   = raw[["Volume"]].copy()

    st.sidebar.caption(f"✅ {len(close_df.columns)} tickers loaded | sample: {list(close_df.columns[:4])}")
except Exception as e:
    st.error(f"Data parsing error: {e}\nRaw columns (first 10): {raw.columns[:10].tolist()}")
    st.stop()

# ── Step 3: technical pre-screen ─────────────────────────────────────────────

passed: list[dict] = []
n = len(universe)

for i, ticker in enumerate(universe):
    prog.progress(35 + int(i / n * 30), text=f"Screening {ticker} ({i+1}/{n})…")

    if ticker not in close_df.columns:
        continue

    px  = close_df[ticker].dropna()
    vol = vol_df[ticker].dropna() if ticker in vol_df.columns else pd.Series(dtype=float)

    if len(px) < 210:
        continue

    last = float(px.iloc[-1])

    # Price filter
    if last < price_min or last > price_max:
        continue

    # Volume filter
    avg_vol = float(vol.tail(30).mean()) if not vol.empty else 0
    if avg_vol < vol_min_k * 1_000:
        continue

    # EMA 200 filter
    ema200 = float(ema(px, 200).iloc[-1])
    if require_above_ema200 and last < ema200:
        continue

    # RSI filter
    r = rsi(px)
    if np.isnan(r) or r < rsi_lo or r > rsi_hi:
        continue

    # HV Percentile filter
    hvp = hv_percentile(px)
    if np.isnan(hvp) or hvp < hvp_min:
        continue

    hv30 = hv_30d(px)

    passed.append({
        "Ticker":      ticker,
        "Price":       round(last, 2),
        "EMA 200":     round(ema200, 2),
        "RSI":         round(r, 1),
        "HV 30d %":    round(hv30, 1),
        "HV Pctile":   hvp,
        "Avg Vol (K)": round(avg_vol / 1_000, 0),
    })

prog.progress(65, text=f"Pre-screen: {len(passed)} candidates. Fetching options chains…")

if not passed:
    prog.progress(100)
    st.warning("No stocks passed the pre-screen. Try relaxing filters (price range, RSI, HVP).")
    st.stop()

# ── Step 4: options chain pull ────────────────────────────────────────────────

final: list[dict] = []

for j, row in enumerate(passed):
    frac = j / len(passed)
    prog.progress(65 + int(frac * 30), text=f"Options: {row['Ticker']} ({j+1}/{len(passed)})…")

    cc = best_cc_strike(
        symbol      = row["Ticker"],
        stock_price = row["Price"],
        dte_min     = dte_min,
        dte_max     = dte_max,
        otm_lo      = otm_lo_pct / 100,
        otm_hi      = otm_hi_pct / 100,
    )

    if cc:
        final.append({**row, **{
            "Expiration":   cc["expiration"],
            "DTE":          cc["dte"],
            "Strike":       cc["strike"],
            "Bid":          cc["bid"],
            "IV %":         cc["iv_pct"],
            "OI":           cc["open_interest"],
            "Ann Yield %":  cc["ann_yield_pct"],
        }})

    time.sleep(0.05)  # gentle rate limiting

prog.progress(100, text="Done!")

# ── Step 5: results ───────────────────────────────────────────────────────────

if not final:
    st.warning(
        "Options screen returned no results. "
        "Try widening the DTE window or OTM % range, or lowering the min bid."
    )
    st.stop()

df = pd.DataFrame(final).sort_values("Ann Yield %", ascending=False).reset_index(drop=True)

st.success(f"✅ **{len(df)} covered call candidates found**  "
           f"(pre-screen: {len(passed)} | options match: {len(df)})")

# Colour-code Ann Yield
def colour_yield(val):
    if val >= 20:
        return "background-color: #1a472a; color: white"
    if val >= 12:
        return "background-color: #2d6a4f; color: white"
    if val >= 6:
        return "background-color: #40916c; color: white"
    return ""

styled = df.style.applymap(colour_yield, subset=["Ann Yield %"])
st.dataframe(styled, use_container_width=True, hide_index=True)

# ── Download ──────────────────────────────────────────────────────────────────

st.download_button(
    label     = "📥 Download CSV",
    data      = df.to_csv(index=False),
    file_name = f"cc_candidates_{datetime.today().strftime('%Y%m%d')}.csv",
    mime      = "text/csv",
)

# ── Disclaimer ────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "⚠️ **Not financial advice.** Data from Yahoo Finance (delayed). "
    "Always verify: (1) no earnings within DTE window, (2) options liquidity in your broker, "
    "(3) tax treatment for your account (LTCG holding period on taxable)."
)
