import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

# Make table rendering resilient to mixed Python object types (Arrow serialization).
def _arrow_safe_frame(frame):
    frame = frame.copy()
    for col in frame.columns:
        series = frame[col]
        if pd.api.types.is_object_dtype(series.dtype):
            non_null = series.dropna()
            value_types = {type(value) for value in non_null}
            has_nested = any(isinstance(value, (list, tuple, dict, set)) for value in non_null)
            if len(value_types) > 1 or has_nested:
                frame[col] = series.map(lambda value: str(value) if value is not None else None).astype("string")
    return frame

def safe_dataframe(data, *args, **kwargs):
    """Display a dataframe; sanitize mixed object columns if Arrow cannot serialize it."""
    try:
        if isinstance(data, pd.DataFrame):
            data = _arrow_safe_frame(data)
        elif hasattr(data, "data") and isinstance(data.data, pd.DataFrame):  # pandas Styler
            original = data
            clean = _arrow_safe_frame(data.data)
            if clean.equals(data.data):
                data = original
            else:
                data = clean  # fallback without Styler formatting for mixed-type columns
        return st.dataframe(data, *args, **kwargs)
    except Exception:
        if hasattr(data, "data") and isinstance(data.data, pd.DataFrame):
            data = data.data
        if isinstance(data, pd.DataFrame):
            return st.dataframe(_arrow_safe_frame(data), *args, **kwargs)
        raise

st.set_page_config(page_title="NSE Terminal Pro", page_icon="📈", layout="wide")

st.title("📈 NSE Terminal")
st.caption("NSE TERMINAL PRO  |  MARKET DATA • TECHNICALS • FUNDAMENTALS • PORTFOLIO INTELLIGENCE")
st.info("Educational tool only. Historical data and simulations cannot guarantee future prices. Not financial advice.")

def ticker(s):
    s=s.strip().upper()
    return s if s.endswith((".NS",".BO")) or s.startswith("^") else s+".NS"

@st.cache_data(ttl=600, show_spinner=False)
def history(s, period="2y"):
    # Explicit timeout prevents the app from appearing blank indefinitely
    # when Yahoo Finance is slow or unreachable.
    return yf.Ticker(ticker(s)).history(
        period=period, auto_adjust=True, timeout=15
    )

@st.cache_data(ttl=10, show_spinner=False)
def latest_intraday_snapshot(s):
    """Latest Yahoo 1-minute candle; Yahoo may delay/update only once per minute."""
    try:
        intraday = yf.Ticker(ticker(s)).history(
            period="1d", interval="1m", auto_adjust=True, timeout=12
        )
        if intraday is not None and not intraday.empty:
            row = intraday.dropna(subset=["Close"]).iloc[-1]
            return {
                "Close": float(row["Close"]),
                "High": float(row["High"]) if pd.notna(row.get("High")) else float(row["Close"]),
                "Low": float(row["Low"]) if pd.notna(row.get("Low")) else float(row["Close"]),
                "Volume": float(row["Volume"]) if pd.notna(row.get("Volume")) else 0.0,
                "Time": str(intraday.dropna(subset=["Close"]).index[-1]),
            }
    except Exception:
        pass
    return None


def latest_intraday_price(s):
    """Best-effort latest intraday quote from Yahoo Finance; may be delayed."""
    try:
        intraday = yf.Ticker(ticker(s)).history(
            period="1d", interval="1m", auto_adjust=False, timeout=12
        )
        if intraday is not None and not intraday.empty:
            prices = intraday["Close"].dropna()
            if not prices.empty:
                return float(prices.iloc[-1])
    except Exception:
        pass
    return None

@st.cache_data(ttl=1800, show_spinner=False)
def fundamentals(s):
    try: return yf.Ticker(ticker(s)).info or {}
