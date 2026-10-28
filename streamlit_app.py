import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(page_title="NSE Terminal", page_icon="📈", layout="wide")
st.markdown("""
<style>
.stApp{background:#0b0b0b;color:#f5f5f5}
.block-container{padding-top:1.2rem;max-width:1450px}
[data-testid="stMetric"]{background:#171717;border:1px solid #303030;padding:12px;border-radius:10px}
</style>
""", unsafe_allow_html=True)
st.title("📈 NSE Terminal")
st.caption("Stock analysis workspace | Clean charts, selectable indicators, forecasting scenarios, risk, watchlist and portfolio")
st.info("Educational tool only. Historical data and simulations cannot guarantee future prices. Not financial advice.")

def ticker(s):
    s=s.strip().upper()
    return s if s.endswith((".NS",".BO")) or s.startswith("^") else s+".NS"

@st.cache_data(ttl=600, show_spinner=False)
def history(s, period="2y"):
    return yf.Ticker(ticker(s)).history(period=period, auto_adjust=True)

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

if not symbol: st.stop()
try: raw=history(symbol,period)
except Exception as e:
    st.error(f"Could not load stock data: {e}"); st.stop()
if raw is None or raw.empty or not {"Close","Open","High","Low"}.issubset(raw.columns):
    st.error("No price data found. Check the ticker symbol."); st.stop()
raw=raw.dropna(subset=["Close"])
if len(raw)<30: st.error("Not enough historical data."); st.stop()
df=add_indicators(raw); close=df.Close; price=float(close.iloc[-1]); ret=close.pct_change().dropna()
info=fundamentals(symbol)

tabs=st.tabs(["Overview","Clean Chart","Indicators","Growth / Loss","Risk","Compare","Watchlist","Portfolio","Fundamentals","News"])

with tabs[0]:
    st.subheader(f"{symbol} overview")
    prev=float(close.iloc[-2]); chg=(price/prev-1)*100 if prev else np.nan
    c=st.columns(4)
    c[0].metric("Latest available price",money(price),f"{chg:+.2f}% vs previous close")
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
    overlays=st.multiselect("Show indicator lines",["EMA 9","EMA 20","EMA 50","SMA 20","SMA 50","SMA 200","BB Upper","BB Middle","BB Lower"],default=["EMA 20","SMA 50"])
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
    chosen=st.selectbox("Choose indicator",["RSI 14","MACD","ADX 14","ATR 14","Daily return (%)","Bollinger Bands"])
    v=df.tail(chart_days); fig=go.Figure()
    if chosen=="MACD":
        fig.add_trace(go.Scatter(x=v.index,y=v["MACD"],name="MACD"))
        fig.add_trace(go.Scatter(x=v.index,y=v["MACD Signal"],name="Signal"))
        fig.add_trace(go.Bar(x=v.index,y=v["MACD Histogram"],name="Histogram"))
    elif chosen=="Bollinger Bands":
        for col in ["BB Upper","BB Middle","BB Lower"]: fig.add_trace(go.Scatter(x=v.index,y=v[col],name=col))
        fig.add_trace(go.Scatter(x=v.index,y=v.Close,name="Close"))
    elif chosen=="Daily return (%)":
        fig.add_trace(go.Bar(x=v.index,y=v.Close.pct_change()*100,name=chosen))
    else:
        fig.add_trace(go.Scatter(x=v.index,y=v[chosen],name=chosen))
        if chosen=="RSI 14": fig.add_hline(y=70,line_dash="dash"); fig.add_hline(y=30,line_dash="dash")
        if chosen=="ADX 14": fig.add_hline(y=25,line_dash="dash")
    fig.update_layout(template="plotly_dark",height=430,title=chosen,hovermode="x unified")
    st.plotly_chart(fig,use_container_width=True)
    cols=["RSI 14","MACD","MACD Signal","MACD Histogram","ADX 14","ATR 14","EMA 9","EMA 20","EMA 50","SMA 20","SMA 50","SMA 200"]
    st.dataframe(pd.DataFrame({"Indicator":cols,"Latest value":[df[c].iloc[-1] for c in cols]}),use_container_width=True,hide_index=True)

with tabs[3]:
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

with tabs[4]:
    st.subheader("Risk analysis")
    vol=float(ret.std()*np.sqrt(252)*100); years=len(ret)/252
    ann=((price/float(close.iloc[0]))**(1/years)-1)*100 if years>0 else np.nan
    dd=close/close.cummax()-1; maxdd=float(dd.min()*100); var=float(np.percentile(ret,5)*100)
    sharpe=(ann-6.5)/vol if vol>0 else np.nan
    c=st.columns(4); c[0].metric("Annualized volatility",f"{vol:.2f}%"); c[1].metric("Annualized historical return",f"{ann:.2f}%"); c[2].metric("Max drawdown",f"{maxdd:.2f}%"); c[3].metric("Daily VaR (95%)",f"{var:.2f}%")
    st.metric("Approx. Sharpe ratio (assumed risk-free rate 6.5%)",f"{sharpe:.2f}" if np.isfinite(sharpe) else "N/A")
    dd_df=pd.DataFrame({"Date":dd.index,"Drawdown (%)":dd.values*100}).tail(chart_days)
    st.plotly_chart(px.area(dd_df,x="Date",y="Drawdown (%)",template="plotly_dark",title="Drawdown from previous peak"),use_container_width=True)

with tabs[5]:
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

with tabs[6]:
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

with tabs[7]:
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

with tabs[8]:
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

with tabs[9]:
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

st.divider()
st.caption(f"Latest price observation: {df.index[-1].date()} | Data: Yahoo Finance via yfinance | Educational use only.")
