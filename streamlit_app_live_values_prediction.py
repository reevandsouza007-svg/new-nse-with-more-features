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
    d["Williams %R"] = -100 * (highest14 - c) / (highest14 - lowest14).replace(0, np.nan)

    # Volume indicators
    direction = np.sign(c.diff()).fillna(0)
    d["OBV"] = (direction * d["Volume"].fillna(0)).cumsum() if "Volume" in d else np.nan
    if "Volume" in d:
        money_flow_mult = ((c - l) - (h - c)) / (h - l).replace(0, np.nan)
        money_flow_vol = money_flow_mult.fillna(0) * d["Volume"].fillna(0)
        positive_flow = money_flow_vol.clip(lower=0).rolling(14).sum()
        negative_flow = (-money_flow_vol.clip(upper=0)).rolling(14).sum()
        d["MFI 14"] = 100 - (100 / (1 + positive_flow / negative_flow.replace(0, np.nan)))
        d["VWAP 20"] = (tp * d["Volume"].fillna(0)).rolling(20).sum() / d["Volume"].fillna(0).rolling(20).sum().replace(0, np.nan)
        d["Volume SMA 20"] = d["Volume"].rolling(20).mean()
    else:
        d["MFI 14"] = np.nan
        d["VWAP 20"] = np.nan
        d["Volume SMA 20"] = np.nan

    # Trend / volatility indicators
    d["Donchian Upper"] = h.rolling(20).max()
    d["Donchian Lower"] = l.rolling(20).min()
    d["Donchian Middle"] = (d["Donchian Upper"] + d["Donchian Lower"]) / 2
    d["Keltner Middle"] = c.ewm(span=20, adjust=False).mean()
    d["Keltner Upper"] = d["Keltner Middle"] + 2 * atr
    d["Keltner Lower"] = d["Keltner Middle"] - 2 * atr

    # Ichimoku lines (unshifted for easy historical display)
    d["Ichimoku Tenkan"] = (h.rolling(9).max() + l.rolling(9).min()) / 2
    d["Ichimoku Kijun"] = (h.rolling(26).max() + l.rolling(26).min()) / 2
    d["Ichimoku Span A"] = (d["Ichimoku Tenkan"] + d["Ichimoku Kijun"]) / 2
    d["Ichimoku Span B"] = (h.rolling(52).max() + l.rolling(52).min()) / 2

    # Supertrend-style direction estimate based on ATR bands
    hl2 = (h + l) / 2
    d["Supertrend Upper"] = hl2 + 3 * atr
    d["Supertrend Lower"] = hl2 - 3 * atr
    d["Daily Return %"] = c.pct_change() * 100
    return d

def money(x):
    try: return f"₹{float(x):,.2f}" if pd.notna(x) and np.isfinite(float(x)) else "N/A"
    except Exception: return "N/A"

with st.sidebar:
    st.header("Stock selector")
    symbol=st.text_input("NSE ticker","RELIANCE").strip().upper()
    period=st.selectbox("Historical data",["6mo","1y","2y","5y","10y"],index=2)
    chart_days=st.selectbox("Chart window",[60,120,252,500,1000],index=2,format_func=lambda n:f"Last {n} trading days")
    st.caption("Examples: RELIANCE, TCS, INFY, HDFCBANK, SBIN")
    st.subheader("🎨 Appearance")
    background_choice = st.selectbox(
        "Choose app background",
        [
            "Default Dark Background",
            "1. Professional Trading Screens",
            "2. Candlestick Charts",
            "3. Futuristic Financial Technology",
            "4. Dark Stock Market Dashboard",
            "5. Bull and Bear Market",
            "6. Minimalist Financial Charts",
        ],
        key="app_background_choice",
    )


# Six selectable stock-market background images.
BACKGROUND_IMAGES = {
    "1. Professional Trading Screens": "https://images.unsplash.com/photo-1611974789855-9c2a0a7236a3?auto=format&fit=crop&w=2200&q=85",
    "2. Candlestick Charts": "https://images.unsplash.com/photo-1642790106117-e829e14a795f?auto=format&fit=crop&w=2200&q=85",
    "3. Futuristic Financial Technology": "https://images.unsplash.com/photo-1639762681485-074b7f938ba0?auto=format&fit=crop&w=2200&q=85",
    "4. Dark Stock Market Dashboard": "https://images.unsplash.com/photo-1590283603385-17ffb3a7f29f?auto=format&fit=crop&w=2200&q=85",
    "5. Bull and Bear Market": "https://images.unsplash.com/photo-1559526324-4b87b5e36e44?auto=format&fit=crop&w=2200&q=85",
    "6. Minimalist Financial Charts": "https://images.unsplash.com/photo-1460925895917-afdab827c52f?auto=format&fit=crop&w=2200&q=85",
}

selected_background = BACKGROUND_IMAGES.get(background_choice)
if selected_background:
    app_background_css = f"""
    .stApp {{
        background-image: linear-gradient(rgba(5, 9, 18, 0.84), rgba(5, 9, 18, 0.90)),
                          url("{selected_background}");
        background-position: center;
        background-size: cover;
        background-attachment: fixed;
        color: #f5f5f5;
    }}
    """
else:
    app_background_css = ".stApp {background: #0b0b0b; color: #f5f5f5;}"

st.markdown(f"""
<style>
{app_background_css}
.block-container {{padding-top: 1.2rem; max-width: 1650px;}}
[data-testid="stAppViewContainer"] {{background-color: rgba(5, 9, 18, 0.10);}}
[data-testid="stSidebar"] {{border-right: 1px solid rgba(255,145,0,0.35);}}
h1, h2, h3 {{letter-spacing: .2px;}}
.stTabs [data-baseweb="tab-list"] {{gap: 5px; border-bottom: 1px solid rgba(255,145,0,.45);}}
.stTabs [data-baseweb="tab"] {{background: rgba(12,18,28,.72); border-radius: 5px 5px 0 0; padding: 8px 12px;}}
.stTabs [aria-selected="true"] {{border-bottom: 2px solid #ff9800; color: #ffb74d;}}
[data-testid="stMetric"] {{border-top: 2px solid rgba(255,152,0,.55);}}
[data-testid="stMetric"] {{
    background: rgba(23, 23, 23, 0.88);
    border: 1px solid rgba(180, 190, 210, 0.25);
    padding: 12px;
    border-radius: 10px;
}}
[data-testid="stSidebar"] {{
    background: rgba(10, 14, 24, 0.94);
}}
</style>
""", unsafe_allow_html=True)

if not symbol: st.stop()
try: raw=history(symbol,period)
except Exception as e:
    st.error(f"Could not load stock data: {e}"); st.stop()
if raw is None or raw.empty or not {"Close","Open","High","Low"}.issubset(raw.columns):
    st.error("No price data found. Check the ticker symbol."); st.stop()
raw=raw.dropna(subset=["Close"])
if len(raw)<30: st.error("Not enough historical data."); st.stop()
df=add_indicators(raw); close=df.Close; daily_close=float(close.iloc[-1]); ret=close.pct_change().dropna()
intraday_snapshot=latest_intraday_snapshot(symbol)
intraday_price=intraday_snapshot["Close"] if intraday_snapshot else None
price=float(intraday_price if intraday_price is not None else daily_close)
price_source="Yahoo Finance intraday quote (may be delayed)" if intraday_price is not None else "Latest available daily close (intraday quote unavailable)"
info=fundamentals(symbol)

def estimate_next_session(hist_df, live_row):
    """Historical-neighbour estimate; descriptive, not a reliable forecast."""
    work = hist_df.copy()
    work["Next Return"] = work["Close"].shift(-1) / work["Close"] - 1
    work["MACD_norm"] = work["MACD Histogram"] / work["Close"].replace(0, np.nan)
    work["EMA_spread"] = (work["EMA 20"] - work["EMA 50"]) / work["Close"].replace(0, np.nan)
    work["Return_5d"] = work["Close"].pct_change(5)
    features = ["RSI 14", "MACD_norm", "EMA_spread", "Return_5d", "ADX 14"]
    train = work[features + ["Next Return"]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(train) < 40:
        return None
    current = pd.Series({
        "RSI 14": live_row.get("RSI 14", np.nan),
        "MACD_norm": live_row.get("MACD Histogram", np.nan) / live_row.get("Close", np.nan),
        "EMA_spread": (live_row.get("EMA 20", np.nan) - live_row.get("EMA 50", np.nan)) / live_row.get("Close", np.nan),
        "Return_5d": hist_df["Close"].pct_change(5).iloc[-1],
        "ADX 14": live_row.get("ADX 14", np.nan),
    })
    if current.isna().any():
        return None
    scale = train[features].std().replace(0, 1).fillna(1)
    distances = (((train[features] - current[features]) / scale) ** 2).sum(axis=1) ** 0.5
    nearest = train.loc[distances.nsmallest(min(30, len(distances))).index, "Next Return"]
    if len(nearest) < 15:
        return None
    probability_up = float((nearest > 0).mean())
    median_return = float(nearest.median())
    if probability_up >= 0.55:
        direction = "UPWARD BIAS"
    elif probability_up <= 0.45:
        direction = "DOWNWARD BIAS"
    else:
        direction = "UNCERTAIN / MIXED"
    return direction, probability_up, median_return, len(nearest)


@st.fragment(run_every="5s")
def live_market_panel():
    st.markdown("## ⚡ Live market panel — updates independently")
    snap = latest_intraday_snapshot(symbol)
    live_price = float(snap["Close"]) if snap else float(price)
    live_df = raw.copy()
    last_idx = live_df.index[-1]
    live_df.loc[last_idx, "Close"] = live_price
    if snap:
        live_df.loc[last_idx, "High"] = max(float(live_df.loc[last_idx, "High"]), snap["High"], live_price)
        live_df.loc[last_idx, "Low"] = min(float(live_df.loc[last_idx, "Low"]), snap["Low"], live_price)
        if "Volume" in live_df.columns:
            existing_volume = live_df.loc[last_idx, "Volume"]
            existing_volume = float(existing_volume) if pd.notna(existing_volume) else 0.0
            live_df.loc[last_idx, "Volume"] = max(existing_volume, snap["Volume"])
    live_ind = add_indicators(live_df)
    last = live_ind.iloc[-1]
    prev_close = float(raw["Close"].iloc[-2]) if len(raw) > 1 else live_price
    change = (live_price / prev_close - 1) * 100 if prev_close else 0.0
    a,b,c,d,e = st.columns(5)
    a.metric("Latest available price", money(live_price), f"{change:+.2f}% vs previous close")
    b.metric("RSI (14)", f"{last['RSI 14']:.2f}" if pd.notna(last["RSI 14"]) else "N/A")
    c.metric("MACD histogram", f"{last['MACD Histogram']:.4f}" if pd.notna(last["MACD Histogram"]) else "N/A")
    d.metric("EMA 20", money(last["EMA 20"]))
    e.metric("ADX (14)", f"{last['ADX 14']:.2f}" if pd.notna(last["ADX 14"]) else "N/A")
    f1,f2,f3,f4,f5 = st.columns(5)
    f1.metric("EMA 50", money(last["EMA 50"]))
    f2.metric("SMA 50", money(last["SMA 50"]))
    f3.metric("ATR (14)", money(last["ATR 14"]))
    f4.metric("Stochastic %K", f"{last['Stochastic %K']:.2f}" if pd.notna(last["Stochastic %K"]) else "N/A")
    f5.metric("ROC (12)", f"{last['ROC 12']:.2f}%" if pd.notna(last["ROC 12"]) else "N/A")
    st.caption("The dashboard does not reload. This panel checks every 5 seconds. Yahoo Finance free intraday data is generally supplied as 1-minute candles and may be delayed, so a new price is shown only when Yahoo publishes an updated value. This is not tick-by-tick or guaranteed real-time data.")
    left, right = st.columns([1.5, 1])
    with left:
        chart = go.Figure()
        recent = live_ind.tail(60)
        chart.add_trace(go.Scatter(x=recent.index, y=recent["Close"], mode="lines", name="Price"))
        chart.add_trace(go.Scatter(x=[last_idx], y=[live_price], mode="markers", name="Latest quote", marker=dict(size=10)))
        chart.update_layout(template="plotly_dark", height=280, margin=dict(l=10,r=10,t=35,b=10), title="Recent price — latest available data")
        st.plotly_chart(chart, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.markdown("### 🔮 Next trading session estimate")
        forecast = estimate_next_session(df, last)
        if forecast:
            direction, probability, median_return, count = forecast
            st.metric("Historical analogue outlook", direction)
            st.metric("Similar historical days positive", f"{probability*100:.1f}%", f"Based on {count} similar days")
            st.metric("Median next-session move in analogues", f"{median_return*100:+.2f}%")
            estimated_price = live_price * (1 + median_return)
            st.metric("Illustrative next-session price", money(estimated_price), f"{median_return*100:+.2f}% analogue move")
        else:
            st.info("Not enough comparable historical observations to estimate a direction.")
        st.caption("This is a simple historical-analogue estimate, not an AI prediction or a reliable guaranteed target. The illustrative price is the current quote multiplied by the median next-session return among similar historical days. Past patterns do not guarantee the next trading session's move.")
    if snap:
        st.caption(f"Latest provider candle timestamp: {snap['Time']} | Source: Yahoo Finance via yfinance (may be delayed).")
    else:
        st.warning("A fresh intraday quote is unavailable. Showing the latest cached/daily price instead.")

live_market_panel()

tabs=st.tabs(["Overview","Clean Chart","Indicators","Strategy Lab","Growth / Loss","Risk","Compare","Watchlist","Portfolio","Fundamentals","News","Backtesting","Stock Scanner","Smart Score","Market Regime","Strategy Leaderboard","Trade Journal","Risk Planner","Swing Trading Signal","Relative Strength Ranking","Walk-Forward Test","Sector Rotation","Price Alert Builder","AI Stock Analyst","Candlestick Patterns","Financial Health Score","Dividend Analysis","Portfolio Optimizer","FII/DII Tracker","Options Analytics","Correlation Matrix","Economic Calendar","AI Research Report"])

with tabs[0]:
    st.subheader(f"{symbol} overview")
    st.caption("For changing values, use the independently updating Live Market Panel above. The metrics below are based on the selected historical daily dataset.")
    prev=float(close.iloc[-1] if intraday_price is not None else close.iloc[-2]); chg=(price/prev-1)*100 if prev else np.nan
    c=st.columns(4)
    c[0].metric("Latest available price",money(price),f"{chg:+.2f}% vs previous close")
    st.caption(f"Price source: {price_source}. Historical overview values; see the Live Market Panel above for the independently updating quote and recalculated indicators.")
    c[1].metric("RSI (14)",f"{df['RSI 14'].iloc[-1]:.2f}" if pd.notna(df["RSI 14"].iloc[-1]) else "N/A")
    c[2].metric("MACD histogram",f"{df['MACD Histogram'].iloc[-1]:.3f}")
    c[3].metric("ADX (14)",f"{df['ADX 14'].iloc[-1]:.2f}" if pd.notna(df["ADX 14"].iloc[-1]) else "N/A")
    votes=sum([df["EMA 20"].iloc[-1]>df["EMA 50"].iloc[-1],df["MACD Histogram"].iloc[-1]>0,
               pd.notna(df["RSI 14"].iloc[-1]) and 50<df["RSI 14"].iloc[-1]<70,
               pd.notna(df["SMA 50"].iloc[-1]) and price>df["SMA 50"].iloc[-1]])
    if votes>=3: st.success("Technical indicators currently lean bullish.")
    elif votes<=1: st.error("Technical indicators currently lean bearish.")
    else: st.warning("Technical indicators are mixed / neutral.")
    st.caption("Indicator summary only; it is not a guaranteed buy/sell recommendation.")

    # Rule-based educational signal and holding-period guidance
    latest = df.iloc[-1]
    rsi_now = latest["RSI 14"]
    macd_bull = bool(latest["MACD"] > latest["MACD Signal"])
    trend_bull = bool(latest["EMA 20"] > latest["EMA 50"])
    price_above_sma50 = bool(pd.notna(latest["SMA 50"]) and price > latest["SMA 50"])
    adx_now = latest["ADX 14"]
    positive_checks = sum([
        macd_bull,
        trend_bull,
        price_above_sma50,
        bool(pd.notna(rsi_now) and 50 <= rsi_now <= 65)
    ])

    st.markdown("### 🧭 Buy / Hold / Avoid signal (rule-based)")
    if positive_checks >= 3 and pd.notna(rsi_now) and rsi_now < 70:
        signal = "WATCH FOR BUY SETUP"
        explanation = "Several indicators are aligned positively. Check support/resistance, volume, company fundamentals and your risk limit before deciding."
        st.success(f"**{signal}** — {explanation}")
    elif positive_checks <= 1 or (pd.notna(rsi_now) and rsi_now >= 70 and not macd_bull):
        signal = "CAUTION / AVOID CHASING"
        explanation = "The selected indicators are weak or mixed. Avoid buying just because the price has recently risen or fallen; wait for clearer confirmation."
        st.warning(f"**{signal}** — {explanation}")
    else:
        signal = "HOLD / WAIT FOR CONFIRMATION"
        explanation = "Signals are mixed. Consider waiting for the trend, MACD and price action to agree rather than making a decision from one indicator."
        st.info(f"**{signal}** — {explanation}")

    st.markdown("### 🗓️ Holding-period guide")
    st.write("This is a general framework, not a prediction of how long this particular stock will rise.")
    h1, h2, h3 = st.columns(3)
    h1.metric("Short-term trade", "Days to 4 weeks", help="Requires close monitoring and a predefined exit/risk plan.")
    h2.metric("Positional trade", "1–6 months", help="Review trend, earnings, valuation and risk regularly.")
    h3.metric("Long-term investing", "3–5+ years", help="Only consider this horizon after checking business quality, valuation, debt, cash flow and your own goal.")
    st.caption("The app cannot reliably tell you an exact number of years to hold. Reassess when the original investment thesis changes, fundamentals weaken, or your goal/risk needs change. RSI/MACD signals can be false and are not guaranteed forecasts.")

    with st.expander("See the signal checks"):
        st.write({
            "RSI (14)": round(float(rsi_now), 2) if pd.notna(rsi_now) else "N/A",
            "MACD above signal": macd_bull,
            "EMA 20 above EMA 50": trend_bull,
            "Price above SMA 50": price_above_sma50,
            "ADX (14)": round(float(adx_now), 2) if pd.notna(adx_now) else "N/A",
            "Positive checks out of 4": positive_checks
        })
    q=st.columns(4)
    q[0].metric("52-week high",money(info.get("fiftyTwoWeekHigh")))
    q[1].metric("52-week low",money(info.get("fiftyTwoWeekLow")))
    q[2].metric("Sector",str(info.get("sector","N/A")))
    q[3].metric("Market cap",f"₹{info['marketCap']/1e7:,.0f} Cr" if info.get("marketCap") else "N/A")

with tabs[1]:
    st.subheader("Clean chart")
    st.write("Choose which lines you want to display on the chart.")
    overlay_options=["EMA 9","EMA 20","EMA 50","SMA 20","SMA 50","SMA 200",
                     "BB Upper","BB Middle","BB Lower","VWAP 20",
                     "Donchian Upper","Donchian Middle","Donchian Lower",
                     "Keltner Upper","Keltner Middle","Keltner Lower",
                     "Ichimoku Tenkan","Ichimoku Kijun","Ichimoku Span A","Ichimoku Span B",
                     "Supertrend Upper","Supertrend Lower"]
    overlays=st.multiselect("Show indicator lines",overlay_options,default=["EMA 20","SMA 50"])
    chart_type=st.radio("Price style",["Candlestick","Closing price line"],horizontal=True)
    volume=st.checkbox("Show volume",False)
    view=df.tail(chart_days); fig=go.Figure()
    if chart_type=="Candlestick":
        fig.add_trace(go.Candlestick(x=view.index,open=view.Open,high=view.High,low=view.Low,close=view.Close,name="Price"))
    else: fig.add_trace(go.Scatter(x=view.index,y=view.Close,mode="lines",name="Close"))
    for col in overlays: fig.add_trace(go.Scatter(x=view.index,y=view[col],mode="lines",name=col))
    if volume and "Volume" in view:
        fig.add_trace(go.Bar(x=view.index,y=view.Volume,name="Volume",yaxis="y2",opacity=.25))
        fig.update_layout(yaxis2=dict(overlaying="y",side="right",showgrid=False))
    fig.update_layout(template="plotly_dark",height=570,title=f"{symbol} price chart",xaxis_rangeslider_visible=False,hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)

with tabs[2]:
    st.subheader("Separate indicator charts")
    chosen=st.selectbox("Choose indicator",[
        "RSI 14","MACD","ADX 14","ATR 14","Daily return (%)","Bollinger Bands",
        "Stochastic","ROC 12","CCI 20","Williams %R","MFI 14","OBV",
        "VWAP 20","Donchian Channels","Keltner Channels","Ichimoku","Volume"
    ])
    v=df.tail(chart_days); fig=go.Figure()
    if chosen=="MACD":
        fig.add_trace(go.Scatter(x=v.index,y=v["MACD"],name="MACD"))
        fig.add_trace(go.Scatter(x=v.index,y=v["MACD Signal"],name="Signal"))
        fig.add_trace(go.Bar(x=v.index,y=v["MACD Histogram"],name="Histogram"))
    elif chosen=="Bollinger Bands":
        for col in ["BB Upper","BB Middle","BB Lower"]: fig.add_trace(go.Scatter(x=v.index,y=v[col],name=col))
        fig.add_trace(go.Scatter(x=v.index,y=v.Close,name="Close"))
    elif chosen=="Daily return (%)":
        fig.add_trace(go.Bar(x=v.index,y=v["Daily Return %"],name=chosen))
    elif chosen=="Stochastic":
        fig.add_trace(go.Scatter(x=v.index,y=v["Stochastic %K"],name="%K"))
        fig.add_trace(go.Scatter(x=v.index,y=v["Stochastic %D"],name="%D"))
        fig.add_hline(y=80,line_dash="dash"); fig.add_hline(y=20,line_dash="dash")
    elif chosen in ["Donchian Channels","Keltner Channels","Ichimoku"]:
        groups = {
            "Donchian Channels":["Donchian Upper","Donchian Middle","Donchian Lower"],
            "Keltner Channels":["Keltner Upper","Keltner Middle","Keltner Lower"],
            "Ichimoku":["Ichimoku Tenkan","Ichimoku Kijun","Ichimoku Span A","Ichimoku Span B"]
        }
        for col in groups[chosen]:
            fig.add_trace(go.Scatter(x=v.index,y=v[col],name=col))
        fig.add_trace(go.Scatter(x=v.index,y=v.Close,name="Close"))
    elif chosen=="Volume":
        if "Volume" in v:
            fig.add_trace(go.Bar(x=v.index,y=v["Volume"],name="Volume"))
            if "Volume SMA 20" in v:
                fig.add_trace(go.Scatter(x=v.index,y=v["Volume SMA 20"],name="Volume SMA 20"))
    else:
        fig.add_trace(go.Scatter(x=v.index,y=v[chosen],name=chosen))
        if chosen=="RSI 14": st.caption("RSI above 70 is often called overbought; below 30 oversold. Neither is an automatic sell/buy signal."); fig.add_hline(y=70,line_dash="dash"); fig.add_hline(y=30,line_dash="dash")
        if chosen=="ADX 14": fig.add_hline(y=25,line_dash="dash")
        if chosen=="Williams %R": fig.add_hline(y=-20,line_dash="dash"); fig.add_hline(y=-80,line_dash="dash")
        if chosen=="MFI 14": fig.add_hline(y=80,line_dash="dash"); fig.add_hline(y=20,line_dash="dash")
    fig.update_layout(template="plotly_dark",height=430,title=chosen,hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)
    cols=["RSI 14","MACD","MACD Signal","MACD Histogram","ADX 14","ATR 14",
          "Stochastic %K","Stochastic %D","ROC 12","CCI 20","Williams %R","MFI 14",
          "OBV","VWAP 20","Volume SMA 20","EMA 9","EMA 20","EMA 50","SMA 20","SMA 50","SMA 200",
          "BB Upper","BB Middle","BB Lower","Donchian Upper","Donchian Middle","Donchian Lower",
          "Keltner Upper","Keltner Middle","Keltner Lower","Ichimoku Tenkan","Ichimoku Kijun",
          "Ichimoku Span A","Ichimoku Span B","Supertrend Upper","Supertrend Lower"]
    st.dataframe(pd.DataFrame({"Indicator":cols,"Latest value":[df[c].iloc[-1] for c in cols]}),use_container_width=True,hide_index=True)

with tabs[3]:
    st.subheader("🧪 Strategy Lab — indicator confluence & pattern radar")
    st.caption("A research workspace that combines indicators, support/resistance and unusual volume. It does not predict with certainty.")
    latest = df.iloc[-1]
    checks = {
        "Trend: EMA 20 > EMA 50": bool(latest["EMA 20"] > latest["EMA 50"]),
        "Price above SMA 50": bool(pd.notna(latest["SMA 50"]) and price > latest["SMA 50"]),
        "MACD bullish": bool(latest["MACD"] > latest["MACD Signal"]),
        "RSI in momentum zone (50–70)": bool(pd.notna(latest["RSI 14"]) and 50 <= latest["RSI 14"] < 70),
        "ADX trend strength > 20": bool(pd.notna(latest["ADX 14"]) and latest["ADX 14"] > 20),
        "ROC positive": bool(pd.notna(latest["ROC 12"]) and latest["ROC 12"] > 0),
        "MFI positive flow (>50)": bool(pd.notna(latest["MFI 14"]) and latest["MFI 14"] > 50),
        "Close above VWAP 20": bool(pd.notna(latest["VWAP 20"]) and price > latest["VWAP 20"]),
    }
    score = sum(checks.values())
    left, right = st.columns([1, 2])
    with left:
        st.metric("Indicator confluence", f"{score}/8")
        st.progress(score / 8)
        if score >= 6:
            st.success("Multiple bullish conditions aligned — research further; not an automatic buy.")
        elif score <= 2:
            st.warning("Few bullish conditions aligned — caution and review the trend.")
        else:
            st.info("Mixed conditions — consider waiting for confirmation.")
    with right:
        st.dataframe(pd.DataFrame([{"Condition": k, "Pass": "Yes" if v else "No"} for k, v in checks.items()]),
                     use_container_width=True, hide_index=True)

    st.markdown("### 📍 Dynamic support and resistance")
    sr_window = st.slider("Look-back window (trading days)", 20, min(120, max(20, len(df)-1)), min(60, max(20, len(df)-1)))
    recent = df.tail(sr_window)
    support = float(recent["Low"].min())
    resistance = float(recent["High"].max())
    a, b, c = st.columns(3)
    a.metric("Latest close", money(price))
    b.metric(f"Approx. support ({sr_window}d)", money(support), f"{(price/support-1)*100:+.2f}% from support" if support else None)
    c.metric(f"Approx. resistance ({sr_window}d)", money(resistance), f"{(resistance/price-1)*100:+.2f}% above price" if price else None)
    st.caption("These are simple recent-window levels, not guaranteed turning points.")

    st.markdown("### ⚡ Unusual-volume radar")
    if "Volume" in df and df["Volume"].notna().sum() > 20:
        vol_latest = latest.get("Volume", np.nan)
        vol_avg = latest.get("Volume SMA 20", np.nan)
        ratio = vol_latest / vol_avg if pd.notna(vol_latest) and pd.notna(vol_avg) and vol_avg else np.nan
        st.metric("Latest volume / 20-day average", f"{ratio:.2f}×" if pd.notna(ratio) else "N/A")
        if pd.notna(ratio) and ratio >= 1.5:
            st.info("Volume is at least 1.5× its 20-day average. Check whether price movement confirms the move.")
        else:
            st.write("No 1.5× volume expansion detected in the latest session, or volume data is unavailable.")
    else:
        st.write("Volume data is unavailable for this ticker/provider.")

    st.markdown("### 🕯️ Simple candle / breakout flags")
    prior_high = df["High"].shift(1).rolling(20).max().iloc[-1]
    prior_low = df["Low"].shift(1).rolling(20).min().iloc[-1]
    flags = [
        ("Close above prior 20-day high", bool(pd.notna(prior_high) and price > prior_high)),
        ("Close below prior 20-day low", bool(pd.notna(prior_low) and price < prior_low)),
        ("RSI above 70 (potentially stretched)", bool(pd.notna(latest["RSI 14"]) and latest["RSI 14"] > 70)),
        ("RSI below 30 (potentially oversold)", bool(pd.notna(latest["RSI 14"]) and latest["RSI 14"] < 30)),
    ]
    st.dataframe(pd.DataFrame([{"Radar flag": k, "Triggered": "Yes" if v else "No"} for k, v in flags]),
                 use_container_width=True, hide_index=True)
    st.caption("This lab uses transparent rules, not machine learning. Backtest and account for transaction costs before using any strategy with real money.")

with tabs[4]:
    st.subheader("Future growth and loss scenarios")
    horizon=st.select_slider("Forecast horizon (trading days)",[7,30,90,252],value=30)
    sims=st.select_slider("Simulation paths",[500,1000,2000,5000],value=1000)
    logs=np.log(close/close.shift()).dropna(); mu=float(logs.mean()); sigma=float(logs.std())
    paths=price*np.exp(np.cumsum(np.random.default_rng(42).normal(mu-.5*sigma**2,sigma,(sims,horizon)),axis=1))
    final=paths[:,-1]; up=(final>price).mean()*100; down=(final<price).mean()*100
    p5,p25,p50,p75,p95=np.percentile(final,[5,25,50,75,95])
    c=st.columns(4); c[0].metric("Current price",money(price)); c[1].metric("Simulated outcomes above",f"{up:.1f}%"); c[2].metric("Simulated outcomes below",f"{down:.1f}%"); c[3].metric("Median scenario",money(p50))
    scenario=pd.DataFrame({"Scenario":["Pessimistic 5th percentile","Lower-middle 25th","Median 50th","Upper-middle 75th","Optimistic 95th percentile"],"Price":[p5,p25,p50,p75,p95]})
    scenario["Change (%)"]=(scenario.Price/price-1)*100
    st.dataframe(scenario.style.format({"Price":"₹{:,.2f}","Change (%)":"{:+.2f}%"}),use_container_width=True,hide_index=True)
    q5,q50,q95=np.percentile(paths,[5,50,95],axis=0); days=np.arange(1,horizon+1); fig=go.Figure()
    fig.add_trace(go.Scatter(x=days,y=q95,name="95th percentile")); fig.add_trace(go.Scatter(x=days,y=q50,name="Median")); fig.add_trace(go.Scatter(x=days,y=q5,name="5th percentile",fill="tonexty"))
    fig.update_layout(template="plotly_dark",height=400,title="Simulated price ranges",xaxis_title="Trading days ahead",yaxis_title="Price (₹)")
    st.plotly_chart(fig,use_container_width=True)
    st.warning("Monte Carlo simulation based on historical returns, not a trained AI forecast. Probabilities are not calibrated real-world odds; actual prices may fall outside these scenarios.")

with tabs[5]:
    st.subheader("Risk analysis")
    vol=float(ret.std()*np.sqrt(252)*100); years=len(ret)/252
    ann=((price/float(close.iloc[0]))**(1/years)-1)*100 if years>0 else np.nan
    dd=close/close.cummax()-1; maxdd=float(dd.min()*100); var=float(np.percentile(ret,5)*100)
    sharpe=(ann-6.5)/vol if vol>0 else np.nan
    c=st.columns(4); c[0].metric("Annualized volatility",f"{vol:.2f}%"); c[1].metric("Annualized historical return",f"{ann:.2f}%"); c[2].metric("Max drawdown",f"{maxdd:.2f}%"); c[3].metric("Daily VaR (95%)",f"{var:.2f}%")
    st.metric("Approx. Sharpe ratio (assumed risk-free rate 6.5%)",f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A")
    dd_df=pd.DataFrame({"Date":dd.index,"Drawdown (%)":dd.values*100}).tail(chart_days)
    st.plotly_chart(px.area(dd_df,x="Date",y="Drawdown (%)",template="plotly_dark",title="Drawdown from previous peak"),use_container_width=True)

with tabs[6]:
    st.subheader("Compare stocks")
    syms=st.text_input("2–5 NSE tickers, comma-separated","RELIANCE,TCS,INFY")
    cp=st.selectbox("Comparison period",["6mo","1y","2y","5y"],index=1)
    if st.button("Compare selected stocks"):
        series={}
        for s in list(dict.fromkeys(x.strip().upper() for x in syms.split(",") if x.strip()))[:5]:
            try:
                h=history(s,cp)
                if not h.empty: series[s]=h.Close/h.Close.dropna().iloc[0]*100
            except Exception: pass
        if len(series)>=2:
            frame=pd.DataFrame(series)
            st.plotly_chart(px.line(frame,template="plotly_dark",title="Relative performance (start=100)"),use_container_width=True)
            st.dataframe(frame.tail(10),use_container_width=True)
        else: st.warning("Could not retrieve at least two valid stock histories.")

with tabs[7]:
    st.subheader("Watchlist")
    watch=st.text_input("Up to 10 tickers","RELIANCE,TCS,INFY,HDFCBANK,SBIN")
    if st.button("Refresh watchlist"):
        rows=[]
        for s in list(dict.fromkeys(x.strip().upper() for x in watch.split(",") if x.strip()))[:10]:
            try:
                h=history(s,"3mo")
                if len(h)>=2:
                    p=float(h.Close.iloc[-1]); prev=float(h.Close.iloc[-2])
                    rows.append({"Symbol":s,"Latest price":p,"Daily change (%)":(p/prev-1)*100,"Date":str(h.index[-1].date())})
            except Exception: pass
        if rows: st.dataframe(pd.DataFrame(rows).style.format({"Latest price":"₹{:,.2f}","Daily change (%)":"{:+.2f}%"}),use_container_width=True,hide_index=True)
        else: st.warning("No watchlist data returned.")

with tabs[8]:
    st.subheader("Portfolio profit/loss calculator")
    st.caption("This calculator does not connect to your broker or save transactions.")
    holdings=[]
    for i in range(3):
        with st.expander(f"Holding {i+1}",expanded=(i==0)):
            a,b,c=st.columns(3)
            s=a.text_input("Ticker",["RELIANCE","TCS","INFY"][i],key=f"ps{i}").strip().upper()
            qty=b.number_input("Quantity",min_value=0.0,value=1.0,key=f"pq{i}")
            buy=c.number_input("Buy price ₹",min_value=0.0,value=100.0,key=f"pb{i}")
            if s and qty>0:
                try:
                    h=history(s,"5d")
                    if not h.empty:
                        now=float(h.Close.iloc[-1]); holdings.append({"Symbol":s,"Qty":qty,"Buy price":buy,"Latest price":now,"Invested":qty*buy,"Value":qty*now,"P/L":qty*(now-buy)})
                except Exception: pass
    if holdings:
        pf=pd.DataFrame(holdings); st.dataframe(pf,use_container_width=True,hide_index=True)
        invested=pf.Invested.sum(); value=pf.Value.sum(); c=st.columns(3)
        c[0].metric("Invested",money(invested)); c[1].metric("Latest value",money(value)); c[2].metric("Profit / loss",money(value-invested),f"{(value/invested-1)*100:+.2f}%" if invested else "N/A")

with tabs[9]:
    st.subheader("Fundamental data")
    fields=[("Company","longName"),("Sector","sector"),("Industry","industry"),("Trailing P/E","trailingPE"),("Forward P/E","forwardPE"),("Price/book","priceToBook"),("EPS","trailingEps"),("ROE","returnOnEquity"),("Profit margin","profitMargins"),("Revenue growth","revenueGrowth"),("Debt/equity","debtToEquity"),("Dividend yield","dividendYield"),("Market cap","marketCap"),("52-week high","fiftyTwoWeekHigh"),("52-week low","fiftyTwoWeekLow")]
    rows=[]
    for label,key in fields:
        val=info.get(key)
        if val is not None:
            if key in ["returnOnEquity","profitMargins","revenueGrowth","dividendYield"]: val=f"{val*100:.2f}%"
            elif key=="marketCap": val=f"₹{val:,.0f}"
            rows.append({"Metric":label,"Value":val})
    if rows: st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    else: st.info("Fundamental information is unavailable for this ticker from the current provider.")

with tabs[10]:
    st.subheader(f"News: {symbol}")
    if st.button("Load latest news"):
        try:
            news=yf.Ticker(ticker(symbol)).news or []
            if not news: st.info("No news returned by provider.")
            for item in news[:15]:
                x=item.get("content",item); title=x.get("title","Untitled")
                publisher=(x.get("provider") or {}).get("displayName") or x.get("publisher") or "Unknown publisher"
                url=(x.get("canonicalUrl") or {}).get("url") or x.get("link")
                with st.expander(title):
                    st.write(f"Publisher: {publisher}")
                    if x.get("pubDate"): st.write(f"Published: {x['pubDate']}")
                    if x.get("summary"): st.write(x["summary"])
                    if url: st.link_button("Open article",url)
        except Exception as e: st.error(f"News could not be loaded: {e}")


with tabs[11]:
    st.subheader("🧪 Strategy Backtesting")
    st.write("Test a rule-based strategy on the selected stock's historical data. Signals are generated from completed-day data and positions are applied from the next trading day to reduce look-ahead bias.")
    if len(df) < 80:
        st.warning("Not enough historical data for a meaningful backtest. Select a longer historical period.")
    else:
        strategy = st.selectbox("Strategy to test", [
            "EMA crossover (20/50)", "RSI mean reversion (30/70)",
            "MACD crossover", "Trend confirmation (EMA + MACD + RSI)"
        ])
        direction = st.selectbox("Trading mode", ["Long only (buy / cash)", "Long / short (simplified)"])
        fee_bps = st.slider("Estimated transaction cost per position change (basis points)", 0, 50, 10, 1,
                            help="10 basis points = 0.10%. This is a simplified cost assumption, not your exact brokerage/taxes.")
        starting_cash = st.number_input("Starting capital (₹)", min_value=1000, max_value=100000000, value=100000, step=10000)

        bt = df.copy().dropna(subset=["Close"])
        close = bt["Close"]
        if strategy == "EMA crossover (20/50)":
            raw = (bt["EMA 20"] > bt["EMA 50"]).astype(float)
        elif strategy == "RSI mean reversion (30/70)":
            # Hold after RSI recovers above 30; exit when RSI reaches 70 or falls below 30.
            raw = pd.Series(np.nan, index=bt.index, dtype=float)
            state = 0.0
            for idx, value in bt["RSI 14"].items():
                if pd.notna(value):
                    if value < 30: state = 1.0
                    elif value > 70: state = 0.0
                raw.loc[idx] = state
            raw = raw.ffill().fillna(0.0)
        elif strategy == "MACD crossover":
            raw = (bt["MACD"] > bt["MACD Signal"]).astype(float)
        else:
            raw = ((bt["EMA 20"] > bt["EMA 50"]) & (bt["MACD"] > bt["MACD Signal"]) & (bt["RSI 14"] > 50)).astype(float)
        if direction == "Long / short (simplified)":
            # Explicitly simplified: short when long signal is off.
            raw = raw * 2 - 1
        # Apply signal on next bar, avoiding same-day close look-ahead.
        position = raw.shift(1).fillna(0.0)
        daily_returns = close.pct_change().fillna(0.0)
        turnover = position.diff().abs().fillna(position.abs())
        cost = turnover * (fee_bps / 10000.0)
        strategy_returns = position * daily_returns - cost
        equity = starting_cash * (1 + strategy_returns).cumprod()
        benchmark = starting_cash * (1 + daily_returns).cumprod()
        peak = equity.cummax()
        drawdown = equity / peak - 1
        total_return = equity.iloc[-1] / starting_cash - 1
        years = max((bt.index[-1] - bt.index[0]).days / 365.25, 1/365.25)
        cagr = (equity.iloc[-1] / starting_cash) ** (1 / years) - 1 if equity.iloc[-1] > 0 else -1
        daily_std = strategy_returns.std()
        sharpe = (strategy_returns.mean() / daily_std * np.sqrt(252)) if daily_std and pd.notna(daily_std) else np.nan
        changes = position.diff().fillna(position.iloc[0]).abs()
        trades = int((changes > 0).sum())
        closed_trade_returns = daily_returns[(position.shift(1).fillna(0) != 0) & (position == 0)]
        win_rate = float((closed_trade_returns > 0).mean() * 100) if len(closed_trade_returns) else np.nan
        b1,b2,b3,b4 = st.columns(4)
        b1.metric("Strategy ending value", money(float(equity.iloc[-1])), f"{total_return*100:+.2f}% total")
        b2.metric("Buy & hold ending value", money(float(benchmark.iloc[-1])), f"{(benchmark.iloc[-1]/starting_cash-1)*100:+.2f}% total")
        b3.metric("Max drawdown", f"{drawdown.min()*100:.2f}%")
        b4.metric("CAGR (approx.)", f"{cagr*100:.2f}%")
        b5,b6,b7 = st.columns(3)
        b5.metric("Sharpe ratio (approx.)", f"{sharpe:.2f}" if pd.notna(sharpe) else "N/A")
        b6.metric("Position changes", f"{trades}")
        b7.metric("Win rate (simple exit-day proxy)", f"{win_rate:.1f}%" if pd.notna(win_rate) else "N/A")
        chart = go.Figure()
        chart.add_trace(go.Scatter(x=equity.index, y=equity, name="Strategy equity"))
        chart.add_trace(go.Scatter(x=benchmark.index, y=benchmark, name="Buy & hold benchmark"))
        chart.update_layout(title="Equity curve", xaxis_title="Date", yaxis_title="Portfolio value (₹)", template="plotly_dark", height=430)
        st.plotly_chart(chart, use_container_width=True)
        ddfig = go.Figure(go.Scatter(x=drawdown.index, y=drawdown*100, name="Drawdown %", fill="tozeroy"))
        ddfig.update_layout(title="Strategy drawdown", xaxis_title="Date", yaxis_title="Drawdown (%)", template="plotly_dark", height=300)
        st.plotly_chart(ddfig, use_container_width=True)
        with st.expander("View latest backtest rows"):
            detail = pd.DataFrame({"Close": close, "Signal before shift": raw, "Position used": position,
                                   "Daily return %": daily_returns*100, "Strategy return %": strategy_returns*100,
                                   "Strategy equity": equity, "Benchmark equity": benchmark, "Drawdown %": drawdown*100})
            st.dataframe(detail.tail(100).round(3), use_container_width=True)
        st.warning("Backtest results are historical simulations, not predictions. They exclude slippage, taxes, liquidity limits, corporate-action complexities and exact brokerage schedules. Long/short mode is simplified and does not model margin or short-selling constraints. Try multiple periods and stocks; avoid selecting a strategy only because it performed well in the past.")



with tabs[12]:
    st.subheader("🔎 Advanced NSE Stock Scanner")
    st.caption("Scans a chosen list of NSE tickers using available Yahoo Finance data. Results are screening signals, not recommendations.")
    default_universe = "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,SBIN,ITC,LT,BHARTIARTL,AXISBANK,TATAMOTORS, SUNPHARMA"
    universe_text = st.text_area("NSE tickers (comma-separated)", default_universe, height=90)
    scan_period = st.selectbox("Scanner history", ["6mo", "1y", "2y"], index=1, key="scan_period")
    max_scan = st.slider("Maximum tickers to scan", 5, 25, 12)
    if st.button("Run stock scan", type="primary"):
        scan_symbols = list(dict.fromkeys(x.strip().upper() for x in universe_text.replace("\n", ",").split(",") if x.strip()))[:max_scan]
        scan_rows=[]
        progress=st.progress(0)
        for i, sym in enumerate(scan_symbols):
            try:
                hist=history(sym, scan_period)
                if hist is None or hist.empty or len(hist)<60:
                    scan_rows.append({"Stock":sym,"Status":"Insufficient data"})
                else:
                    ind=add_indicators(hist.dropna(subset=["Close"]))
                    z=ind.iloc[-1]
                    close_now=float(z["Close"])
                    conditions=[pd.notna(z.get("EMA 20")) and z["EMA 20"]>z["EMA 50"],
                                pd.notna(z.get("MACD")) and z["MACD"]>z["MACD Signal"],
                                pd.notna(z.get("RSI 14")) and 50<=z["RSI 14"]<=70,
                                pd.notna(z.get("SMA 50")) and close_now>z["SMA 50"],
                                pd.notna(z.get("ROC 12")) and z["ROC 12"]>0,
                                pd.notna(z.get("ADX 14")) and z["ADX 14"]>=20]
                    score=int(sum(bool(x) for x in conditions))
                    label="Bullish conditions" if score>=5 else ("Mixed conditions" if score>=3 else "Caution")
                    scan_rows.append({"Stock":sym,"Price (₹)":round(close_now,2),"Daily change %":round(float(ind.Close.pct_change().iloc[-1]*100),2),
                                      "RSI":round(float(z["RSI 14"]),2) if pd.notna(z["RSI 14"]) else np.nan,
                                      "ADX":round(float(z["ADX 14"]),2) if pd.notna(z["ADX 14"]) else np.nan,
                                      "Conditions passed / 6":score,"Assessment":label,"Status":"OK"})
            except Exception as e:
                scan_rows.append({"Stock":sym,"Status":f"Data error: {str(e)[:70]}"})
            progress.progress((i+1)/max(1,len(scan_symbols)))
        scan_df=pd.DataFrame(scan_rows)
        if "Conditions passed / 6" in scan_df:
            scan_df=scan_df.sort_values("Conditions passed / 6",ascending=False,na_position="last")
        st.session_state["nse_scan_results"]=scan_df
    if "nse_scan_results" in st.session_state:
        scan_df=st.session_state["nse_scan_results"]
        st.dataframe(scan_df,use_container_width=True,hide_index=True)
        st.download_button("Download scan results CSV",scan_df.to_csv(index=False).encode("utf-8"),"nse_stock_scan.csv","text/csv")
    st.warning("Yahoo Finance data can be delayed, incomplete or unavailable. A high score does not guarantee future gains. Confirm ticker data and test any rule historically.")

with tabs[13]:
    st.subheader("🧠 Fundamental + Technical Smart Score")
    st.caption("Transparent, rule-based scoring. Missing fundamentals are excluded rather than treated as good or bad.")
    latest=df.iloc[-1]
    tech_checks={
        "EMA 20 above EMA 50":pd.notna(latest["EMA 20"]) and latest["EMA 20"]>latest["EMA 50"],
        "Price above SMA 50":pd.notna(latest["SMA 50"]) and price>latest["SMA 50"],
        "MACD above signal":pd.notna(latest["MACD"]) and latest["MACD"]>latest["MACD Signal"],
        "RSI between 50 and 70":pd.notna(latest["RSI 14"]) and 50<=latest["RSI 14"]<=70,
        "Positive 12-period ROC":pd.notna(latest.get("ROC 12")) and latest["ROC 12"]>0,
        "ADX at least 20":pd.notna(latest["ADX 14"]) and latest["ADX 14"]>=20,
    }
    tech_score=100*sum(bool(v) for v in tech_checks.values())/len(tech_checks)
    st.metric("Technical conditions score",f"{tech_score:.0f}/100")
    st.dataframe(pd.DataFrame([{"Technical condition":k,"Pass":"Yes" if v else "No"} for k,v in tech_checks.items()]),use_container_width=True,hide_index=True)
    st.markdown("#### Fundamental checks")
    fundamental_rules=[("Revenue growth", "revenueGrowth", lambda x:x>0, "Positive revenue growth"),
                       ("Profit margin", "profitMargins", lambda x:x>0.08, "Profit margin above 8%"),
                       ("Return on equity", "returnOnEquity", lambda x:x>0.12, "ROE above 12%"),
                       ("Debt/equity", "debtToEquity", lambda x:x<100, "Debt/equity below 100 (provider units)"),
                       ("Free cash flow", "freeCashflow", lambda x:x>0, "Positive free cash flow")]
    frows=[]
    for label,key,rule,description in fundamental_rules:
        val=info.get(key)
        try:
            numeric=float(val) if val is not None else np.nan
            passed=bool(pd.notna(numeric) and rule(numeric))
            frows.append({"Metric":label,"Available value":round(numeric*100,2) if key in ["revenueGrowth","profitMargins","returnOnEquity"] and pd.notna(numeric) else (round(numeric,2) if pd.notna(numeric) else "Unavailable"),"Rule":description,"Result":"Pass" if passed else ("Fail" if pd.notna(numeric) else "Missing")})
        except Exception:
            frows.append({"Metric":label,"Available value":"Unavailable","Rule":description,"Result":"Missing"})
    st.dataframe(pd.DataFrame(frows),use_container_width=True,hide_index=True)
    available=[r for r in frows if r["Result"]!="Missing"]
    fund_score=100*sum(r["Result"]=="Pass" for r in available)/len(available) if available else np.nan
    a,b,c=st.columns(3)
    a.metric("Technical score",f"{tech_score:.0f}/100")
    b.metric("Fundamental score",f"{fund_score:.0f}/100" if pd.notna(fund_score) else "N/A")
    combined=(tech_score+fund_score)/2 if pd.notna(fund_score) else np.nan
    c.metric("Combined score",f"{combined:.0f}/100" if pd.notna(combined) else "N/A")
    st.caption("Weights are illustrative and equal-weighted. Provider fundamentals may be missing, stale or reported in different units; verify figures against company filings before making decisions.")

with tabs[14]:
    st.subheader("🌦️ Market Regime Detector")
    st.caption("Estimates trend and volatility conditions for the selected stock and Nifty 50 using historical price data.")
    regime_period=st.selectbox("Regime lookback",["6mo","1y","2y","5y"],index=1,key="regime_period")
    regime_rows=[]
    for name, sym in [(symbol,symbol),("NIFTY 50","^NSEI")]:
        try:
            h=history(sym,regime_period)
            if h is None or h.empty or len(h)<40: continue
            h=add_indicators(h.dropna(subset=["Close"]))
            q=h.iloc[-1]
            vol=float(h.Close.pct_change().rolling(20).std().iloc[-1]*np.sqrt(252)*100)
            trend_up=bool(pd.notna(q.get("SMA 50")) and q.Close>q["SMA 50"] and q["EMA 20"]>q["EMA 50"])
            trend_down=bool(pd.notna(q.get("SMA 50")) and q.Close<q["SMA 50"] and q["EMA 20"]<q["EMA 50"])
            regime="Bullish trend" if trend_up else ("Bearish trend" if trend_down else "Sideways / mixed")
            if pd.notna(vol) and vol>35: regime += " + High volatility"
            regime_rows.append({"Instrument":name,"Latest close":round(float(q.Close),2),"50-day SMA":round(float(q["SMA 50"]),2) if pd.notna(q["SMA 50"]) else np.nan,"Annualized 20-day volatility %":round(vol,2),"Estimated regime":regime})
        except Exception as e:
            regime_rows.append({"Instrument":name,"Estimated regime":f"Unavailable: {str(e)[:60]}"})
    if regime_rows: st.dataframe(pd.DataFrame(regime_rows),use_container_width=True,hide_index=True)
    else: st.info("Market data unavailable for this period.")
    st.caption("Regime labels are simple rule-based classifications, not official market states or predictions.")

with tabs[15]:
    st.subheader("🏆 Strategy Leaderboard")
    st.caption(f"Compares simple strategy rules on {symbol} using the selected historical dataset. Positions are shifted to the next bar to reduce look-ahead bias.")
    start_cash=st.number_input("Leaderboard starting capital (₹)",min_value=1000,max_value=100000000,value=100000,step=10000,key="leader_cash")
    fee=st.slider("Cost per position change (basis points)",0,50,10,key="leader_fee")/10000
    close_series=df.Close.dropna()
    daily=close_series.pct_change().fillna(0)
    strategy_signals={
        "EMA 20/50":(df["EMA 20"]>df["EMA 50"]).astype(float),
        "MACD crossover":(df["MACD"]>df["MACD Signal"]).astype(float),
        "RSI momentum":((df["RSI 14"]>50)&(df["RSI 14"]<70)&(df["EMA 20"]>df["EMA 50"])).astype(float),
        "Trend + MACD + RSI":((df["EMA 20"]>df["EMA 50"])&(df["MACD"]>df["MACD Signal"])&(df["RSI 14"]>50)).astype(float),
    }
    leaderboard=[]
    for name,sig in strategy_signals.items():
        pos=sig.shift(1).fillna(0)
        strat_ret=pos*daily-pos.diff().abs().fillna(pos.abs())*fee
        eq=start_cash*(1+strat_ret).cumprod()
        dd=(eq/eq.cummax()-1).min()*100
        std=strat_ret.std()
        sharpe=strat_ret.mean()/std*np.sqrt(252) if pd.notna(std) and std>0 else np.nan
        leaderboard.append({"Strategy":name,"Total return %":round((eq.iloc[-1]/start_cash-1)*100,2),"Max drawdown %":round(float(dd),2),"Sharpe (approx.)":round(float(sharpe),2) if pd.notna(sharpe) else np.nan,"Position changes":int((pos.diff().fillna(pos.iloc[0]).abs()>0).sum())})
    lead=pd.DataFrame(leaderboard).sort_values("Sharpe (approx.)",ascending=False,na_position="last")
    st.dataframe(lead,use_container_width=True,hide_index=True)
    st.download_button("Download leaderboard CSV",lead.to_csv(index=False).encode("utf-8"),"strategy_leaderboard.csv","text/csv")
    st.warning("This is an in-sample comparison, so it may overfit historical data. Validate on a separate time period and include slippage, taxes, liquidity and corporate actions before real-world use.")

with tabs[16]:
    st.subheader("📓 Trade Journal")
    st.caption("Record paper trades or your own trades and review your decision process. Entries are stored in this active Streamlit session; download CSV to keep a permanent copy.")
    if "trade_journal_rows" not in st.session_state:
        st.session_state["trade_journal_rows"]=[]
    with st.form("trade_journal_form",clear_on_submit=True):
        j1,j2,j3=st.columns(3)
        j_symbol=j1.text_input("Ticker",value=symbol).strip().upper()
        j_side=j2.selectbox("Side",["BUY","SELL"])
        j_date=j3.date_input("Trade date")
        j4,j5,j6=st.columns(3)
        j_entry=j4.number_input("Entry price (₹)",min_value=0.0,value=float(price),step=1.0)
        j_qty=j5.number_input("Quantity",min_value=1,value=1,step=1)
        j_exit=j6.number_input("Exit price (₹; 0 if open)",min_value=0.0,value=0.0,step=1.0)
        j_reason=st.text_input("Reason / setup")
        j_notes=st.text_area("Notes / lessons learned")
        submitted=st.form_submit_button("Add trade to journal")
    if submitted:
        if not j_symbol or j_entry<=0:
            st.error("Enter a ticker and a valid entry price.")
        else:
            pnl=(j_exit-j_entry)*j_qty if j_exit>0 else np.nan
            if j_side=="SELL" and pd.notna(pnl): pnl=-pnl
            st.session_state["trade_journal_rows"].append({"Date":str(j_date),"Ticker":j_symbol,"Side":j_side,"Entry ₹":j_entry,"Quantity":int(j_qty),"Exit ₹":j_exit if j_exit>0 else "Open","Realized P/L ₹":round(pnl,2) if pd.notna(pnl) else "Open","Reason":j_reason,"Notes":j_notes})
            st.success("Trade added to this session's journal.")
    journal=pd.DataFrame(st.session_state["trade_journal_rows"])
    if not journal.empty:
        st.dataframe(journal,use_container_width=True,hide_index=True)
        st.download_button("Download trade journal CSV",journal.to_csv(index=False).encode("utf-8"),"trade_journal.csv","text/csv")
        if st.button("Clear current session journal"):
            st.session_state["trade_journal_rows"]=[]
            st.rerun()
    else:
        st.info("No trades recorded yet. Add a trade above or use paper trades to practise.")

with tabs[17]:
    st.subheader("🎯 Risk Planner: position size, stop-loss and target")
    st.caption("A planning calculator only. It does not identify a guaranteed entry point.")
    p1,p2,p3=st.columns(3)
    capital=p1.number_input("Account capital (₹)",min_value=1000.0,value=100000.0,step=5000.0)
    risk_pct=p2.slider("Maximum planned risk per trade (%)",0.25,5.0,1.0,0.25)
    entry=p3.number_input("Hypothetical entry price (₹)",min_value=0.01,value=float(price),step=1.0)
    p4,p5=st.columns(2)
    stop_pct=p4.slider("Stop-loss distance (%)",0.5,20.0,5.0,0.5)
    reward_risk=p5.selectbox("Target reward:risk ratio",[1.0,1.5,2.0,3.0],index=2)
    risk_budget=capital*risk_pct/100
    stop=entry*(1-stop_pct/100)
    per_share_risk=entry-stop
    qty=max(0,int(risk_budget/per_share_risk)) if per_share_risk>0 else 0
    target=entry+per_share_risk*reward_risk
    r1,r2,r3,r4=st.columns(4)
    r1.metric("Planned risk budget",money(risk_budget))
    r2.metric("Illustrative stop-loss",money(stop))
    r3.metric("Illustrative target",money(target))
    r4.metric("Position size (shares)",str(qty))
    st.write(f"Approximate position value: **{money(qty*entry)}**. Approximate loss at stop, before costs/gaps: **{money(qty*per_share_risk)}**.")
    st.warning("Stop orders may execute worse than the stop price during gaps or illiquid conditions. Position sizing does not remove investment risk.")

st.divider()
st.caption(f"Latest price observation: {df.index[-1].date()} | Data: Yahoo Finance via yfinance | Educational use only.")


with tabs[18]:
    st.subheader("📅 Swing Trading Signal — hold for days to weeks")
    st.caption("Rule-based research for a multi-day holding period. It cannot know tomorrow's price or guarantee a profitable trade.")

    latest = df.iloc[-1]
    prev_close = float(df["Close"].iloc[-2])
    current_close = float(latest["Close"])
    rsi_now = latest.get("RSI 14", np.nan)
    macd_hist = latest.get("MACD Histogram", np.nan)
    ema20 = latest.get("EMA 20", np.nan)
    ema50 = latest.get("EMA 50", np.nan)
    sma50 = latest.get("SMA 50", np.nan)
    adx_now = latest.get("ADX 14", np.nan)

    # ATR for illustrative risk levels
    tr = pd.concat([
        df["High"] - df["Low"],
        (df["High"] - df["Close"].shift()).abs(),
        (df["Low"] - df["Close"].shift()).abs()
    ], axis=1).max(axis=1)
    atr_now = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean().iloc[-1]

    bullish = {
        "EMA 20 above EMA 50": bool(pd.notna(ema20) and pd.notna(ema50) and ema20 > ema50),
        "Close above SMA 50": bool(pd.notna(sma50) and current_close > sma50),
        "MACD histogram positive": bool(pd.notna(macd_hist) and macd_hist > 0),
        "RSI between 50 and 70": bool(pd.notna(rsi_now) and 50 <= rsi_now <= 70),
        "ADX above 20": bool(pd.notna(adx_now) and adx_now > 20),
    }
    score = sum(bullish.values())
    if score >= 4 and pd.notna(rsi_now) and rsi_now < 70:
        signal = "POTENTIAL SWING BUY SETUP"
        message = "Several bullish conditions align. Check liquidity, news, market conditions and risk/reward before acting."
        st.success(f"**{signal}** — {message}")
    elif score <= 1 or (pd.notna(rsi_now) and rsi_now > 75):
        signal = "CAUTION / AVOID CHASING"
        message = "The setup is weak or potentially stretched. Waiting for clearer confirmation may be prudent."
        st.warning(f"**{signal}** — {message}")
    else:
        signal = "WAIT FOR CONFIRMATION"
        message = "The indicators are mixed. There is no strong alignment under these rules."
        st.info(f"**{signal}** — {message}")

    st.markdown("### Signal details")
    st.write(f"Latest available candle: **{df.index[-1].strftime('%Y-%m-%d')}**")
    st.write(f"Bullish conditions met: **{score}/5**")
    st.dataframe(pd.DataFrame([{"Condition": k, "Status": "Met" if v else "Not met"} for k, v in bullish.items()]),
                 use_container_width=True, hide_index=True)

    st.markdown("### Illustrative trade planning")
    risk_pct = st.slider("Risk budget (% of trading capital)", min_value=0.25, max_value=2.0, value=1.0, step=0.25)
    capital = st.number_input("Trading capital (₹)", min_value=1000.0, value=100000.0, step=5000.0)
    stop_atr = st.slider("Stop-loss distance (ATR multiple)", min_value=1.0, max_value=3.0, value=1.5, step=0.25)
    target_multiple = st.slider("Target risk/reward multiple", min_value=1.0, max_value=4.0, value=2.0, step=0.5)
    if pd.notna(atr_now) and atr_now > 0:
        stop_price = max(0.01, current_close - stop_atr * float(atr_now))
        risk_per_share = current_close - stop_price
        target_price = current_close + target_multiple * risk_per_share
        risk_budget = capital * risk_pct / 100
        shares_by_risk = int(risk_budget // risk_per_share) if risk_per_share > 0 else 0
        shares_by_cash = int(capital // current_close) if current_close > 0 else 0
        shares = max(0, min(shares_by_risk, shares_by_cash))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reference price", f"₹{current_close:,.2f}")
        c2.metric("Illustrative stop-loss", f"₹{stop_price:,.2f}", f"-{(current_close-stop_price)/current_close*100:.2f}%")
        c3.metric("Illustrative target", f"₹{target_price:,.2f}", f"+{(target_price/current_close-1)*100:.2f}%")
        c4.metric("Estimated quantity", f"{shares:,} shares")
        st.caption(f"Risk budget: ₹{risk_budget:,.2f}. Quantity is limited by both risk budget and available capital; actual fills, gaps, costs and slippage can change outcomes.")
    else:
        st.warning("Not enough data to calculate ATR-based planning levels.")

    st.markdown("### Next trading session outlook")
    st.caption("The Live Market Panel above also shows a historical-analogue estimate based on similar past indicator conditions.")
    st.write("**Outlook:** " + ("Bullish conditions may continue if price and volume confirm." if score >= 4 else "Mixed/uncertain; wait for the next completed candle to confirm." if score in [2,3] else "Bearish or weak conditions dominate under this rule set."))
    st.caption("This is a conditional outlook, not a prediction of tomorrow's price. Signals use historical daily data and may be delayed or incomplete during market hours.")



with tabs[19]:
    st.subheader("📈 Relative Strength Ranking")
    st.caption("Ranks a chosen list by historical performance versus the Nifty 50. This is backward-looking, not a prediction.")
    rs_text = st.text_area("Tickers to compare (comma-separated)", "RELIANCE,TCS,INFY,HDFCBANK,ICICIBANK,SBIN,ITC,LT,BHARTIARTL,SUNPHARMA", key="rs_universe")
    if st.button("Rank stocks", key="rank_rs"):
        tickers = list(dict.fromkeys(x.strip().upper() for x in rs_text.split(",") if x.strip()))[:30]
        try:
            benchmark = history("^NSEI", "1y")
            if benchmark.empty:
                st.warning("Nifty 50 benchmark data is unavailable right now.")
            else:
                bclose = benchmark["Close"].dropna()
                rows=[]
                for sym in tickers:
                    try:
                        hist = history(sym, "1y")
                        c = hist["Close"].dropna()
                        if len(c) < 25: continue
                        row={"Ticker":sym,"Latest price":float(c.iloc[-1])}
                        for label,days in [("1M return %",21),("3M return %",63),("6M return %",126)]:
                            row[label]=float((c.iloc[-1]/c.iloc[-min(days+1,len(c))]-1)*100)
                            bdays=min(days+1,len(bclose))
                            bret=float((bclose.iloc[-1]/bclose.iloc[-bdays]-1)*100)
                            row[label.replace("return","relative vs Nifty")]=row[label]-bret
                        rows.append(row)
                    except Exception:
                        continue
                if rows:
                    result=pd.DataFrame(rows).sort_values("3M relative vs Nifty %",ascending=False)
                    st.dataframe(result.round(2),use_container_width=True,hide_index=True)
                    st.download_button("Download ranking CSV",result.to_csv(index=False).encode(),"relative_strength_ranking.csv","text/csv")
                else: st.warning("No valid ticker data returned. Check ticker symbols or try again later.")
        except Exception as e: st.error(f"Could not retrieve benchmark data: {e}")

with tabs[20]:
    st.subheader("🧪 Walk-Forward Test")
    st.caption("A simple chronological split: build/inspect a rule on the earlier period, then evaluate the same fixed rule on later data. This is not an optimizer and does not guarantee future performance.")
    split_pct=st.slider("Earlier period used for development (%)",50,85,70,5,key="wf_split")
    wf_strategy=st.selectbox("Strategy",["EMA crossover (20/50)","MACD crossover","Trend confirmation (EMA + MACD + RSI)"],key="wf_strategy")
    wf_fee=st.slider("Estimated cost per position change (basis points)",0,50,10,key="wf_fee")
    wdf=df.dropna(subset=["Close"]).copy()
    split=max(30,min(len(wdf)-10,int(len(wdf)*split_pct/100)))
    if len(wdf)<80:
        st.warning("Select at least 80 trading days for this basic split test.")
    else:
        if wf_strategy=="EMA crossover (20/50)": sig=(wdf["EMA 20"]>wdf["EMA 50"]).astype(float)
        elif wf_strategy=="MACD crossover": sig=(wdf["MACD"]>wdf["MACD Signal"]).astype(float)
        else: sig=((wdf["EMA 20"]>wdf["EMA 50"])&(wdf["MACD"]>wdf["MACD Signal"])&(wdf["RSI 14"]>50)).astype(float)
        pos=sig.shift(1).fillna(0)
        daily=wdf["Close"].pct_change().fillna(0)
        strat=pos*daily-pos.diff().abs().fillna(pos.abs())*(wf_fee/10000)
        test_idx=wdf.index[split:]
        test_returns=strat.loc[test_idx]
        bench_returns=daily.loc[test_idx]
        strat_eq=(1+test_returns).cumprod()
        bench_eq=(1+bench_returns).cumprod()
        test_dd=(strat_eq/strat_eq.cummax()-1).min()*100
        st.write(f"Development period: **{wdf.index[0].date()} to {wdf.index[split-1].date()}**")
        st.write(f"Test period: **{wdf.index[split].date()} to {wdf.index[-1].date()}**")
        m1,m2,m3=st.columns(3)
        m1.metric("Strategy test return",f"{(strat_eq.iloc[-1]-1)*100:+.2f}%")
        m2.metric("Buy & hold test return",f"{(bench_eq.iloc[-1]-1)*100:+.2f}%")
        m3.metric("Test-period max drawdown",f"{test_dd:.2f}%")
        fig=go.Figure()
        fig.add_trace(go.Scatter(x=strat_eq.index,y=(strat_eq-1)*100,name="Strategy test return %"))
        fig.add_trace(go.Scatter(x=bench_eq.index,y=(bench_eq-1)*100,name="Buy & hold test return %"))
        fig.update_layout(title="Out-of-sample test period",xaxis_title="Date",yaxis_title="Cumulative return (%)",template="plotly_dark",height=400)
        st.plotly_chart(fig,use_container_width=True)
        st.warning("This is a single holdout split, not full rolling walk-forward validation. Test more stocks and periods, and include realistic slippage, taxes and execution rules.")

with tabs[21]:
    st.subheader("🌐 Sector Rotation Dashboard")
    st.caption("Attempts to compare selected Yahoo Finance sector-index symbols. Some index symbols may be unavailable; missing data is reported rather than fabricated.")
    sectors={"Nifty Bank":"^NSEBANK","Nifty IT":"^CNXIT","Nifty Pharma":"^CNXPHARMA","Nifty Auto":"^CNXAUTO","Nifty Metal":"^CNXMETAL","Nifty Energy":"^CNXENERGY","Nifty FMCG":"^CNXFMCG","Nifty Realty":"^CNXREALTY"}
    if st.button("Load sector performance",key="load_sector_perf"):
        sector_rows=[]
        progress=st.progress(0)
        for i,(name,sym) in enumerate(sectors.items()):
            try:
                hist=history(sym,"6mo")
                c=hist["Close"].dropna()
                if len(c)>22:
                    sector_rows.append({"Sector index":name,"Symbol":sym,"1M return %":(c.iloc[-1]/c.iloc[-min(22,len(c))]-1)*100,"3M return %":(c.iloc[-1]/c.iloc[-min(64,len(c))]-1)*100})
            except Exception: pass
            progress.progress((i+1)/len(sectors))
        if sector_rows:
            sdf=pd.DataFrame(sector_rows).sort_values("3M return %",ascending=False)
            st.dataframe(sdf.round(2),use_container_width=True,hide_index=True)
            fig=px.bar(sdf.sort_values("3M return %"),x="3M return %",y="Sector index",orientation="h",title="Sector index performance over approx. 3 months")
            fig.update_layout(template="plotly_dark",height=430)
            st.plotly_chart(fig,use_container_width=True)
            st.download_button("Download sector data",sdf.to_csv(index=False).encode(),"sector_rotation.csv","text/csv")
        else: st.warning("No sector-index data was returned. Provider symbols may change or be unsupported.")

with tabs[22]:
    st.subheader("🔔 Price Alert Builder")
    st.caption("Builds a threshold alert for this session. Streamlit Community Cloud does not send push notifications from this widget; the alert view checks when its tab is rerun; it is not a background push-notification service.")
    alert_kind=st.selectbox("Alert condition",["Price rises above level","Price falls below level","RSI crosses above 50 zone (current check)","RSI falls below 30 zone (current check)"],key="alert_kind")
    threshold=st.number_input("Price threshold (₹)",min_value=0.0,value=float(round(price,2)),step=1.0,key="alert_threshold")
    if alert_kind=="Price rises above level": triggered=price>threshold
    elif alert_kind=="Price falls below level": triggered=price<threshold
    elif alert_kind=="RSI crosses above 50 zone (current check)": triggered=pd.notna(df["RSI 14"].iloc[-1]) and df["RSI 14"].iloc[-1]>50
    else: triggered=pd.notna(df["RSI 14"].iloc[-1]) and df["RSI 14"].iloc[-1]<30
    st.metric("Latest available price",money(price))
    st.metric("Latest RSI (14)",f"{df['RSI 14'].iloc[-1]:.2f}" if pd.notna(df["RSI 14"].iloc[-1]) else "N/A")
    if triggered: st.success("ALERT CONDITION MET on the latest available data.")
    else: st.info("Alert condition is not met on the latest available data.")
    st.caption(f"Latest available candle: {df.index[-1].date()}. Data may be delayed; this is not a continuously running alert service.")

# ─────────────────────────────────────────────────────────────────────────────
# NEW RESEARCH & FINTECH MODULES (educational decision-support; provider limits
# are disclosed rather than presenting unavailable data as live facts).
# ─────────────────────────────────────────────────────────────────────────────

with tabs[23]:
    st.subheader("🤖 AI Stock Analyst")
    st.caption("Explainable, rule-based analyst assistant. It does not use a large-language model or predict guaranteed returns.")
    last = df.iloc[-1]
    rsi_now = float(last["RSI 14"]) if pd.notna(last["RSI 14"]) else np.nan
    macd_now = float(last["MACD Histogram"]) if pd.notna(last["MACD Histogram"]) else 0.0
    trend_up = bool(last["EMA 20"] > last["EMA 50"]) if pd.notna(last["EMA 20"]) and pd.notna(last["EMA 50"]) else False
    above_sma = bool(price > last["SMA 50"]) if pd.notna(last["SMA 50"]) else False
    adx_now = float(last["ADX 14"]) if pd.notna(last["ADX 14"]) else np.nan
    analyst_score = sum([trend_up, macd_now > 0, above_sma, pd.notna(rsi_now) and 50 <= rsi_now <= 70, pd.notna(adx_now) and adx_now >= 20])
    a,b,c = st.columns(3)
    a.metric("Technical score", f"{analyst_score}/5")
    b.metric("Trend", "Uptrend" if trend_up else "Not confirmed")
    c.metric("Momentum", "Positive" if macd_now > 0 else "Negative / flat")
    if analyst_score >= 4:
        st.success("Rule-based view: bullish technical setup. Check valuation, event risk and position sizing before any decision.")
    elif analyst_score <= 1:
        st.error("Rule-based view: weak technical setup. Avoid treating a low score as a short-sale instruction.")
    else:
        st.warning("Rule-based view: mixed signals. Consider waiting for clearer confirmation.")
    st.markdown("**Why this assessment?**")
    st.write({"EMA 20 above EMA 50": trend_up, "MACD histogram positive": macd_now > 0,
              "Price above SMA 50": above_sma, "RSI in 50–70 zone": bool(pd.notna(rsi_now) and 50 <= rsi_now <= 70),
              "ADX at least 20": bool(pd.notna(adx_now) and adx_now >= 20)})
    st.info("This is transparent indicator-based decision support, not personalised investment advice or a true generative-AI forecast.")

with tabs[24]:
    st.subheader("🕯️ Candlestick Pattern Recognition")
    st.caption("Scans the latest daily candles for basic textbook patterns. Patterns need context and are not standalone trade signals.")
    candle = df.dropna(subset=["Open","High","Low","Close"]).copy()
    candle["Body"] = (candle["Close"]-candle["Open"]).abs()
    candle["Range"] = (candle["High"]-candle["Low"]).replace(0,np.nan)
    candle["Upper wick"] = candle["High"]-candle[["Open","Close"]].max(axis=1)
    candle["Lower wick"] = candle[["Open","Close"]].min(axis=1)-candle["Low"]
    if len(candle) >= 2:
        x, prevc = candle.iloc[-1], candle.iloc[-2]
        patterns=[]
        if x["Body"]/x["Range"] < 0.10: patterns.append("Doji: open and close are very close")
        if x["Lower wick"] >= 2*max(x["Body"],0.000001) and x["Upper wick"] <= max(x["Body"],0.000001): patterns.append("Hammer-like candle: long lower wick")
        if x["Upper wick"] >= 2*max(x["Body"],0.000001) and x["Lower wick"] <= max(x["Body"],0.000001): patterns.append("Shooting-star-like candle: long upper wick")
        bullish_engulf = (prevc["Close"] < prevc["Open"] and x["Close"] > x["Open"] and x["Open"] <= prevc["Close"] and x["Close"] >= prevc["Open"])
        bearish_engulf = (prevc["Close"] > prevc["Open"] and x["Close"] < x["Open"] and x["Open"] >= prevc["Close"] and x["Close"] <= prevc["Open"])
        if bullish_engulf: patterns.append("Bullish engulfing: current real body covers prior body")
        if bearish_engulf: patterns.append("Bearish engulfing: current real body covers prior body")
        if not patterns: st.info("No selected basic pattern detected in the latest candle.")
        else:
            for pattern in patterns: st.success(pattern)
        st.write(f"Latest candle date: **{candle.index[-1].date()}**")
        st.dataframe(candle.tail(10)[["Open","High","Low","Close","Body","Upper wick","Lower wick"]].round(2),use_container_width=True)
    else: st.warning("Not enough candles to scan.")

with tabs[25]:
    st.subheader("🏥 Financial Health Score")
    st.caption("A simple 0–100 screening score using available Yahoo Finance fundamentals. Missing fields are excluded; it is not a credit rating.")
    f=info or {}
    checks=[("Positive profit margin", f.get("profitMargins"), lambda v:v>0),
            ("Positive return on equity", f.get("returnOnEquity"), lambda v:v>0),
            ("Debt/equity below 1", f.get("debtToEquity"), lambda v:v<100),
            ("Positive operating cash flow", f.get("operatingCashflow"), lambda v:v>0),
            ("Positive revenue growth", f.get("revenueGrowth"), lambda v:v>0),
            ("Current ratio above 1", f.get("currentRatio"), lambda v:v>1)]
    known=[(name,val,fn) for name,val,fn in checks if val is not None and pd.notna(val)]
    if known:
        passed=sum(bool(fn(float(val))) for _,val,fn in known)
        score=round(passed/len(known)*100)
        st.metric("Available-data health score",f"{score}/100",f"{passed} of {len(known)} checks passed")
        rows=[]
        for name,val,fn in known: rows.append({"Test":name,"Reported value":val,"Result":"Pass" if fn(float(val)) else "Needs attention"})
        st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)
    else: st.warning("Fundamental fields are unavailable for this ticker from the current data provider.")
    st.caption("Data can be incomplete, stale or mapped differently for Indian listings. Verify figures from company filings.")

with tabs[26]:
    st.subheader("💰 Dividend Analysis")
    try:
        div=yf.Ticker(ticker(symbol)).dividends
        if div is None or div.empty: st.info("No dividend history was returned by Yahoo Finance for this ticker.")
        else:
            div=div.sort_index()
            d1,d2=st.columns(2)
            d1.metric("Latest dividend",money(div.iloc[-1]))
            d2.metric("Total dividends in returned history",money(div.sum()))
            st.line_chart(div.rename("Dividend per share"),use_container_width=True)
            annual=div.resample("YE").sum().rename("Annual dividend per share").to_frame()
            st.dataframe(annual.tail(10).round(3),use_container_width=True)
            st.caption("Dividend yield and payout depend on current price, tax and corporate actions. Historical dividends do not guarantee future payments.")
    except Exception as e: st.warning(f"Dividend data could not be loaded: {e}")

with tabs[27]:
    st.subheader("⚖️ Portfolio Optimizer")
    st.caption("Illustrative long-only allocation using inverse historical volatility. This is a risk-balancing heuristic, not a guaranteed optimal portfolio.")
    opt_text=st.text_input("Enter 3–10 NSE tickers separated by commas",f"{symbol}, TCS, INFY",key="opt_tickers")
    if st.button("Calculate illustrative allocation",key="run_port_opt"):
        syms=list(dict.fromkeys([x.strip().upper() for x in opt_text.split(",") if x.strip()]))[:10]
        returns={}
        with st.spinner("Loading price histories..."):
            for sym in syms:
                try:
                    hst=history(sym,"2y")["Close"].dropna()
                    r=hst.pct_change().dropna()
                    if len(r)>60: returns[sym]=r
                except Exception: pass
        if len(returns)<2: st.warning("At least two tickers with sufficient historical data are required.")
        else:
            ret_df=pd.DataFrame(returns).dropna()
            vol=ret_df.std()*np.sqrt(252)
            inv=1/vol.replace(0,np.nan)
            weights=(inv/inv.sum()).dropna()
            out=pd.DataFrame({"Ticker":weights.index,"Illustrative weight %":weights.values*100,"Annualised historical volatility %":vol[weights.index].values*100})
            st.dataframe(out.round(2),use_container_width=True,hide_index=True)
            fig=px.pie(out,names="Ticker",values="Illustrative weight %",hole=.45,title="Inverse-volatility allocation")
            fig.update_layout(template="plotly_dark")
            st.plotly_chart(fig,use_container_width=True)
            st.caption("Uses aligned daily returns and inverse volatility only. It does not account for expected returns, correlations in the weighting rule, liquidity, taxes or your personal risk profile.")

with tabs[28]:
    st.subheader("🏦 FII / DII Activity Tracker")
    st.caption("Enter figures from an official NSE/NSDL/CDSL source or upload a CSV. This app does not currently fetch a verified live FII/DII feed.")
    uploaded=st.file_uploader("Upload CSV with Date, FII Net (₹ Cr), DII Net (₹ Cr)",type=["csv"],key="fiidii_csv")
    if uploaded is not None:
        try:
            flow=pd.read_csv(uploaded)
            required=["Date","FII Net (₹ Cr)","DII Net (₹ Cr)"]
            if all(col in flow.columns for col in required):
                flow["Date"]=pd.to_datetime(flow["Date"],errors="coerce")
                flow=flow.dropna(subset=["Date"]).sort_values("Date")
                st.dataframe(flow,use_container_width=True,hide_index=True)
                if not flow.empty:
                    st.metric("Latest combined net flow",f"₹{(flow.iloc[-1]['FII Net (₹ Cr)']+flow.iloc[-1]['DII Net (₹ Cr)']):,.2f} Cr")
                    st.line_chart(flow.set_index("Date")[["FII Net (₹ Cr)","DII Net (₹ Cr)"]])
            else: st.error("CSV needs these exact columns: Date, FII Net (₹ Cr), DII Net (₹ Cr).")
        except Exception as e: st.error(f"Could not read CSV: {e}")
    else:
        st.info("No file uploaded. Use the sample template below to prepare your data.")
        sample=pd.DataFrame([{"Date":"2026-11-03","FII Net (₹ Cr)":-500,"DII Net (₹ Cr)":650}])
        st.download_button("Download CSV template",sample.to_csv(index=False).encode(),"fiidii_template.csv","text/csv")

with tabs[29]:
    st.subheader("📊 Options Analytics")
    st.caption("Attempts to retrieve option-chain data through Yahoo Finance. NSE option-chain coverage may be unavailable or incomplete for many symbols.")
    try:
        opt_ticker=yf.Ticker(ticker(symbol))
        expiries=opt_ticker.options
        if not expiries: st.warning("No options expiries are available from this data provider for the selected ticker.")
        else:
            expiry=st.selectbox("Expiry",expiries,key="option_expiry")
            chain=opt_ticker.option_chain(expiry)
            calls=chain.calls.copy(); puts=chain.puts.copy()
            call_oi=float(calls["openInterest"].fillna(0).sum()) if "openInterest" in calls else 0
            put_oi=float(puts["openInterest"].fillna(0).sum()) if "openInterest" in puts else 0
            pcr=put_oi/call_oi if call_oi else np.nan
            q1,q2,q3=st.columns(3)
            q1.metric("Call open interest",f"{call_oi:,.0f}")
            q2.metric("Put open interest",f"{put_oi:,.0f}")
            q3.metric("Put/Call OI ratio",f"{pcr:.2f}" if pd.notna(pcr) else "N/A")
            st.write("Calls")
            st.dataframe(calls.head(30),use_container_width=True,hide_index=True)
            st.write("Puts")
            st.dataframe(puts.head(30),use_container_width=True,hide_index=True)
    except Exception as e: st.warning(f"Options data is not available for this ticker through Yahoo Finance: {e}")
    st.info("PCR is descriptive, not a directional guarantee. Verify Indian index/stock option-chain data with an exchange-authorised source before use.")

with tabs[30]:
    st.subheader("🔗 Stock Correlation Matrix")
    corr_text=st.text_input("Tickers separated by commas",f"{symbol}, TCS, INFY, HDFCBANK, ICICIBANK",key="corr_tickers")
    if st.button("Build correlation matrix",key="build_corr"):
        data={}
        for sym in list(dict.fromkeys([x.strip().upper() for x in corr_text.split(",") if x.strip()]))[:12]:
            try:
                hst=history(sym,"1y")["Close"].dropna()
                if len(hst)>30: data[sym]=hst.pct_change()
            except Exception: pass
        if len(data)>=2:
            corr=pd.DataFrame(data).dropna().corr()
            fig=px.imshow(corr,zmin=-1,zmax=1,color_continuous_scale="RdYlGn",text_auto=".2f",title="Daily-return correlation")
            fig.update_layout(template="plotly_dark",height=600)
            st.plotly_chart(fig,use_container_width=True)
            st.dataframe(corr.round(3),use_container_width=True)
        else: st.warning("At least two tickers with overlapping historical data are required.")
    st.caption("Correlation measures how returns moved together historically. It does not establish causation or guarantee future diversification.")

with tabs[31]:
    st.subheader("🗓️ Economic Calendar")
    st.caption("A personal event tracker. Events below are entered by you; this is not a verified live economic calendar.")
    with st.form("economic_event_form"):
        event_date=st.date_input("Event date")
        event_name=st.text_input("Event name",placeholder="RBI policy / CPI / GDP release")
        event_impact=st.selectbox("Potential impact",["High","Medium","Low"])
        event_notes=st.text_input("Notes / official source")
        submitted=st.form_submit_button("Add event")
    if submitted and event_name.strip():
        st.session_state.setdefault("economic_events",[]).append({"Date":str(event_date),"Event":event_name.strip(),"Impact":event_impact,"Notes":event_notes})
    events=pd.DataFrame(st.session_state.get("economic_events",[]))
    if not events.empty:
        st.dataframe(events.sort_values("Date"),use_container_width=True,hide_index=True)
        st.download_button("Download events CSV",events.to_csv(index=False).encode(),"economic_calendar.csv","text/csv")
    else: st.info("No events added yet. Add RBI, inflation, GDP, budget or company-result dates and record their official source.")

with tabs[32]:
    st.subheader("📝 AI Research Report")
    st.caption("Creates a structured, explainable draft from available price indicators and Yahoo Finance fundamentals. It is automated template analysis, not a generative-AI opinion or verified analyst report.")
    f=info or {}
    report_date=pd.Timestamp.now().strftime("%d %b %Y")
    report_rsi=df["RSI 14"].iloc[-1]
    report_macd=df["MACD Histogram"].iloc[-1]
    report_trend="positive" if df["EMA 20"].iloc[-1]>df["EMA 50"].iloc[-1] else "not confirmed positive"
    st.markdown(f"### {symbol} — Equity Research Snapshot")
    st.write(f"**Generated:** {report_date}  |  **Latest available price:** {money(price)}  |  **Data date:** {df.index[-1].date()}")
    st.markdown("**1. Business and valuation data**")
    st.write(f"Company: {f.get('longName',symbol)}. Sector: {f.get('sector','Not available')}. Industry: {f.get('industry','Not available')}.")
    st.write(f"Market capitalisation: {money(f.get('marketCap'))}. Trailing P/E: {f.get('trailingPE','Not available')}. Forward P/E: {f.get('forwardPE','Not available')}.")
    st.markdown("**2. Technical view**")
    st.write(f"The short-term EMA trend is {report_trend}. RSI (14): {report_rsi:.2f} if available; MACD histogram: {report_macd:.3f}.")
    st.markdown("**3. Key risks and limitations**")
    st.write("Market volatility, earnings changes, valuation risk, sector conditions, liquidity and data-provider delays should be reviewed. This report does not include a verified management assessment or forward cash-flow valuation.")
    st.markdown("**4. Analyst conclusion**")
    st.write("Use the technical signals as a screening starting point only. Verify the latest annual report, quarterly results, exchange disclosures and valuation assumptions before forming an investment view.")
    report_text=(f"{symbol} — Equity Research Snapshot\nGenerated: {report_date}\nLatest available price: {money(price)}\nCompany: {f.get('longName',symbol)}\nSector: {f.get('sector','Not available')}\nMarket cap: {money(f.get('marketCap'))}\nTrailing P/E: {f.get('trailingPE','Not available')}\nForward P/E: {f.get('forwardPE','Not available')}\nRSI 14: {report_rsi}\nMACD histogram: {report_macd}\nEMA trend: {report_trend}\nLimitations: Data may be delayed/incomplete. Verify official filings. Educational use only; not investment advice.")
    st.download_button("Download research snapshot TXT",report_text.encode(),f"{symbol.lower()}_research_snapshot.txt","text/plain")