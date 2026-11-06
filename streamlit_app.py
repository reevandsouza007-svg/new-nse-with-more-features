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
