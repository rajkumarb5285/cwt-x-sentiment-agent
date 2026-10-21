"""Live dashboard:  streamlit run app.py
Auto-runs a refresh cycle each bar (hourly by default) and re-draws forecasts, charts and the summary."""
import json
import os
import time
import pandas as pd
import streamlit as st
from src import db, viz
from src.config import load_config, has_live_keys
from src.pipeline import Pipeline

st.set_page_config(page_title="CWT X Sentiment Agent", layout="wide")
cfg = load_config()
OUT = cfg["paths"]["outputs"]

st.sidebar.title("X Sentiment Agent")
default_live = has_live_keys()
mode = st.sidebar.radio("Data mode", ["live", "demo"], index=0 if default_live else 1,
                        help="demo = synthetic data, works offline. live = Apify + Binance + Tavily (needs .env).")
auto = st.sidebar.checkbox("Auto-refresh each bar", value=True)
if mode == "demo":
    st.sidebar.warning("Demo mode: SYNTHETIC data.")
if mode == "live" and not default_live:
    st.sidebar.error("Add APIFY_TOKEN and TAVILY_API_KEY to .env")

if "last_run" not in st.session_state:
    st.session_state.last_run = 0.0


def run_cycle(first=False):
    with st.spinner("Running pipeline (ingest, sentiment, features, forecast)..."):
        Pipeline(cfg, mode, full_eval=first and not os.path.exists(os.path.join(OUT, "metrics.json"))).run(verbose=False)
    st.session_state.last_run = time.time()


if st.sidebar.button("Run cycle now", type="primary"):
    run_cycle()

interval = cfg["live"]["interval_minutes"] * 60
if not os.path.exists(os.path.join(OUT, "forecast_latest.csv")):
    run_cycle(first=True)


@st.fragment(run_every="60s")
def dashboard():
    if auto and mode == "live" and time.time() - st.session_state.last_run > interval:
        run_cycle()
    df = pd.read_csv(os.path.join(OUT, "features.csv"), index_col=0, parse_dates=True)
    fc = pd.read_csv(os.path.join(OUT, "forecast_latest.csv"), parse_dates=["made_at", "target_ts"])
    with open(os.path.join(OUT, "metrics.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    bt = meta["backtest"]
    last = df.iloc[-1]
    f0, f5 = fc.iloc[0], fc.iloc[-1]

    st.title(f"{cfg['asset']['name']} X sentiment agent")
    st.caption(f"{'SYNTHETIC DEMO DATA' if meta.get('mode') == 'demo' else 'Live data'} | last closed bar "
               f"{df.index[-1]:%Y-%m-%d %H:%M} UTC | refreshed {time.strftime('%H:%M:%S')}")
    c = st.columns(5)
    c[0].metric("Last close", f"${last.close:,.0f}")
    c[1].metric("Forecast n+0", f"${f0.pred_price:,.0f}", f"{f0.pred_ret:+.2%}")
    c[2].metric("Forecast n+5", f"${f5.pred_price:,.0f}", f"{f5.pred_price / f5.base_close - 1:+.2%}")
    c[3].metric("Sentiment (24h)", f"{df.sent_w.iloc[-24:].mean():+.2f}")
    c[4].metric("Post volume z", f"{df.vol_z.iloc[-24:].mean():+.1f}")

    st.plotly_chart(viz.fig_forecast(df, fc), use_container_width=True)
    st.plotly_chart(viz.fig_sentiment_vs_price(df), use_container_width=True)

    t1, t2, t3, t4 = st.tabs(["Summary", "Performance", "Live scorecard", "Forecast table"])
    with t1:
        with open(os.path.join(OUT, "sentiment_summary.md"), encoding="utf-8") as fh:
            st.markdown(fh.read())
    with t2:
        s, b = bt["strategy"], bt["buy_hold"]
        st.dataframe(pd.DataFrame({"Strategy": s, "Buy & hold": b}).loc[
            ["sharpe", "sortino", "max_drawdown", "total_return", "win_rate"]].style.format("{:.3f}"))
        bf = pd.read_csv(os.path.join(OUT, "backtest_frame.csv"), index_col=0, parse_dates=True)
        st.plotly_chart(viz.fig_equity(bf), use_container_width=True)
        st.dataframe(pd.read_csv(os.path.join(OUT, "horizon_metrics.csv")))
    with t3:
        conn = db.connect(cfg["paths"]["db"] if mode == "live" else cfg["paths"]["db"].replace(".db", "_demo.db"))
        f_all, px = db.read(conn, "forecasts", "made_at"), db.read(conn, "prices")
        if len(f_all):
            m = f_all.merge(px[["ts", "close"]], left_on="target_ts", right_on="ts", how="left").dropna(subset=["close"])
            m["error_pct"] = (m.pred_price / m.close - 1) * 100
            st.caption("Past forecasts vs prices that have since been realised.")
            st.dataframe(m[["made_at", "target_ts", "horizon", "pred_price", "close", "error_pct"]].tail(60)
                         .rename(columns={"close": "realised_close"}), use_container_width=True)
        else:
            st.info("No realised forecasts yet. They appear after the first predicted bars close.")
    with t4:
        st.dataframe(fc, use_container_width=True)


dashboard()
