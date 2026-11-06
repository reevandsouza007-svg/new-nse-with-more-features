import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="NSE Terminal Pro", page_icon="📈", layout="wide")

st.title("📈 NSE Terminal")
st.caption("NSE TERMINAL PRO  |  MARKET DATA • TECHNICALS • FUNDAMENTALS • PORTFOLIO INTELLIGENCE")
st.info("Educational tool only. Historical data and simulations cannot guarantee future prices. Not financial advice.")

def ticker(s):
    s=s.strip().upper()
    return s if s.endswith((".NS",".BO")) or s.startswith("^") else s+".NS"

@st.cache_data(ttl=600, show_spinner=False)
def history(s, period="2y"):
    return yf.Ticker(ticker(s)).history(period=period, auto_adjust=True)

@st.cache_data(ttl=10, show_spinner=False)
def latest_intraday_snapshot(s):
    """Latest Yahoo 1-minute candle; Yahoo may delay/update only once per minute."""
    try:
        intraday = yf.Ticker(ticker(s)).history(period="1d", interval="1m", auto_adjust=True)
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
        intraday = yf.Ticker(ticker(s)).history(period="1d", interval="1m", auto_adjust=False)
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
    except Exception: return {}

def rsi(series, n=14):
    d=series.diff()
    gain=d.clip(lower=0).ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    loss=(-d.clip(upper=0)).ewm(alpha=1/n,min_periods=n,adjust=False).mean()
    rs=gain/loss.replace(0,np.nan)
    return 100-100/(1+rs)

def add_indicators(df):
    d=df.copy(); c=d.Close; h=d.High; l=d.Low
    for n in [9,20,50]: d[f"EMA {n}"]=c.ewm(span=n,adjust=False).mean()
    for n in [20,50,200]: d[f"SMA {n}"]=c.rolling(n).mean()
    d["RSI 14"]=rsi(c)
    d["MACD"]=c.ewm(span=12,adjust=False).mean()-c.ewm(span=26,adjust=False).mean()
    d["MACD Signal"]=d["MACD"].ewm(span=9,adjust=False).mean()
    d["MACD Histogram"]=d["MACD"]-d["MACD Signal"]
    mid=c.rolling(20).mean(); sd=c.rolling(20).std()
    d["BB Upper"]=mid+2*sd; d["BB Middle"]=mid; d["BB Lower"]=mid-2*sd
    tr=pd.concat([h-l,(h-c.shift()).abs(),(l-c.shift()).abs()],axis=1).max(axis=1)
    atr=tr.ewm(alpha=1/14,min_periods=14,adjust=False).mean()
    up=h.diff(); down=-l.diff()
    plus=up.where((up>down)&(up>0),0.0); minus=down.where((down>up)&(down>0),0.0)
    pdi=100*plus.ewm(alpha=1/14,min_periods=14,adjust=False).mean()/atr
    mdi=100*minus.ewm(alpha=1/14,min_periods=14,adjust=False).mean()/atr
    dx=100*(pdi-mdi).abs()/(pdi+mdi).replace(0,np.nan)
    d["ADX 14"]=dx.ewm(alpha=1/14,min_periods=14,adjust=False).mean()
    d["ATR 14"]=atr

    # Momentum indicators
    d["ROC 12"] = c.pct_change(12) * 100
    low14 = l.rolling(14).min()
    high14 = h.rolling(14).max()
    d["Stochastic %K"] = 100 * (c - low14) / (high14 - low14).replace(0, np.nan)
    d["Stochastic %D"] = d["Stochastic %K"].rolling(3).mean()
    tp = (h + l + c) / 3
    sma_tp = tp.rolling(20).mean()
    mad = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
    d["CCI 20"] = (tp - sma_tp) / (0.015 * mad.replace(0, np.nan))
    highest14 = h.rolling(14).max()
    lowest14 = l.rolling(14).min()
