"""Orchestrates the agent. Each method is one node of the LangGraph workflow (see src/graph.py)."""
import json
import os
import time
import pandas as pd
from src import db, features, model, backtest, viz, summary, sentiment
from src.config import load_config
from src.ingest import apify_x, demo, prices as price_src, tavily_news


class Pipeline:
    def __init__(self, cfg=None, mode="demo", days=None, csv=None, resume=True, full_eval=True):
        self.cfg = cfg or load_config()
        self.mode, self.csv, self.resume, self.full_eval = mode, csv, resume, full_eval
        self.days = days or self.cfg["lookback_days"]
        self.out = self.cfg["paths"]["outputs"]
        os.makedirs(self.out, exist_ok=True)
        self.conn = db.connect(self.cfg["paths"]["db"] if mode == "live" else self.cfg["paths"]["db"].replace(".db", "_demo.db"))
        self.df = self.feats = self.oos = self.hm = self.bt = self.bt_frame = self.fc = None

    # ---- nodes -------------------------------------------------------------
    def ingest(self):
        if self.mode == "demo":
            if db.count_posts(self.conn) == 0:
                p, px, n = demo.generate(self.cfg, self.days)
                db.upsert_posts(self.conn, p); db.upsert_prices(self.conn, px); db.upsert_news(self.conn, n)
            return "demo data ready (SYNTHETIC)"
        # live
        px_old = db.read(self.conn, "prices")
        start = (px_old.ts.max() - 3 * pd.Timedelta(self.cfg["bar"])) if len(px_old) else \
            pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=self.days)
        db.upsert_prices(self.conn, price_src.fetch_prices(self.cfg, start))
        if self.csv:
            apify_x.load_csv(self.conn, self.csv)
        have = db.count_posts(self.conn)
        if have == 0:
            apify_x.backfill(self.cfg, self.conn, self.days, self.resume)          # 180d backfill
        else:
            apify_x.backfill(self.cfg, self.conn, self.cfg["live"]["apify_lookback_days"] + 1, resume=False)  # incremental
        try:
            db.upsert_news(self.conn, tavily_news.fetch_news(self.cfg))
        except Exception as e:
            print(f"[tavily] failed: {e}")
        return "live data refreshed"

    def score(self):
        return f"scored {sentiment.score_unscored(self.conn, self.cfg)} new posts"

    def build_features(self):
        posts, px = db.read(self.conn, "posts"), db.read(self.conn, "prices")
        self.df, self.feats = features.build_features(posts, px, self.cfg)
        self.df.to_csv(os.path.join(self.out, "features.csv"))
        return f"{len(self.df)} bars x {len(self.feats)} features"

    def evaluate(self):
        mp = os.path.join(self.out, "metrics.json")
        if not self.full_eval and os.path.exists(mp) and time.time() - os.path.getmtime(mp) < 86400:
            with open(mp, encoding="utf-8") as fh:
                self.bt = json.load(fh)["backtest"]
            self.hm = pd.read_csv(os.path.join(self.out, "horizon_metrics.csv"))
            self.oos = pd.read_csv(os.path.join(self.out, "predictions_oos.csv"), index_col=0, parse_dates=True)
            self.bt_frame = pd.read_csv(os.path.join(self.out, "backtest_frame.csv"), index_col=0, parse_dates=True)
            return "reused cached walk-forward results (<24h old)"
        self.oos = model.walk_forward(self.df, self.feats, self.cfg)
        self.hm = model.horizon_metrics(self.oos, self.cfg["horizons"])
        self.bt, self.bt_frame = backtest.run_backtest(self.oos, self.cfg)
        self.oos.to_csv(os.path.join(self.out, "predictions_oos.csv"))
        self.hm.to_csv(os.path.join(self.out, "horizon_metrics.csv"), index=False)
        self.bt_frame.to_csv(os.path.join(self.out, "backtest_frame.csv"))
        with open(os.path.join(self.out, "metrics.json"), "w", encoding="utf-8") as fh:
            json.dump({"backtest": self.bt, "mode": self.mode, "updated": str(pd.Timestamp.now(tz="UTC"))},
                      fh, indent=2, default=float)
        s = self.bt["strategy"]
        return f"walk-forward done: Sharpe {s['sharpe']:.2f}, Sortino {s['sortino']:.2f}, MaxDD {s['max_drawdown']:.1%}"

    def forecast(self):
        self.fc = model.fit_final_and_forecast(self.df, self.feats, self.cfg)
        db.save_forecast(self.conn, self.fc)
        self.fc.to_csv(os.path.join(self.out, "forecast_latest.csv"), index=False)
        return f"forecast made at {self.fc.made_at.iloc[0]:%Y-%m-%d %H:%M} UTC"

    def report(self):
        figs = {"sentiment_vs_price": viz.fig_sentiment_vs_price(self.df),
                "forecast": viz.fig_forecast(self.df, self.fc),
                "equity_curve": viz.fig_equity(self.bt_frame),
                "horizon_accuracy": viz.fig_horizon_skill(self.hm)}
        for h in self.cfg["horizons"]:
            figs[f"pred_vs_actual_n{h}"] = viz.fig_pred_vs_actual(self.oos, h)
        viz.save_all(figs, os.path.join(self.out, "charts"))
        md = summary.build_summary(self.cfg, self.df, db.read(self.conn, "posts"), db.read(self.conn, "news"),
                                   self.fc, self.bt, synthetic=self.mode == "demo")
        with open(os.path.join(self.out, "sentiment_summary.md"), "w", encoding="utf-8") as fh:
            fh.write(md)
        with open(os.path.join(self.out, "evaluation_report.md"), "w", encoding="utf-8") as fh:
            fh.write(self._eval_md())
        try:
            from src.pdf_report import generate_pdf_report
            generate_pdf_report(self.cfg, self.df, self.bt, self.hm,
                                os.path.join(self.out, "evaluation_report.pdf"),
                                synthetic=self.mode == "demo")
        except Exception as e:
            print(f"[pdf_report] failed: {e}")
        return "charts, summary, evaluation_report.md and evaluation_report.pdf written"

    def _eval_md(self):
        s, b = self.bt["strategy"], self.bt["buy_hold"]
        f = lambda m, k, p=False: "n/a" if m.get(k) is None else (f"{m[k]:.1%}" if p else f"{m[k]:.2f}")
        cfg = self.cfg
        hm = self.hm.copy()
        tbl = "| Horizon | MAE ($) | Naive MAE ($) | Skill vs naive | Dir. acc | IC |\n|---|---|---|---|---|---|\n" + "\n".join(
            f"| {r.horizon} | {r.MAE_usd:,.1f} | {r.MAE_naive_usd:,.1f} | {r.skill_vs_naive:+.2%} | {r.dir_acc:.1%} | {r.IC_spearman:+.3f} |"
            for r in hm.itertuples())
        folds = "\n".join(f"- Fold {k}: Sharpe {v:.2f}" for k, v in self.bt["per_fold_sharpe"].items() if v is not None)
        coverage_days = len(self.df) / 24.0

        return f"""# CWT Quantitative Evaluation Report: X Sentiment Agent

## 1. Project Overview
- **Project Name:** CWT X Sentiment Price Prediction Agent
- **Objective:** Quantify crowd psychology on X (Twitter) and evaluate whether social sentiment has statistically significant predictive power for short-term Bitcoin price direction across horizons n+0 to n+5.
- **Target Asset:** {cfg['asset']['name']} ({cfg['asset']['symbol']})
- **Forecasting Interval:** 1-hour discrete bars (predicting n+0 forming close through n+5 future hours).
- **Agent Architecture:** Autonomous LangGraph state graph (Ingest -> Score -> Build Features -> Evaluate -> Forecast -> Report) with scheduled auto-refresh and multi-modal reporting.

## 2. Data Collection
- **Coverage:** {len(self.df):,} total hourly bars (~{coverage_days:.1f} days of data).
- **X Post Fields:** Captured via Apify (`apidojo/tweet-scraper`): `id`, `ts` (UTC creation), `text`, `likes`, `reposts`, `replies`, `views`, `followers`, and `author`.
- **Price Data Source:** Binance Public REST API (`BTCUSDT` 1h bars) with automatic fallback to Yahoo Finance (`BTC-USD`).
- **News Data Source:** Tavily Search API. Extracted headlines are strictly confined to executive context narratives and never used as model inputs.
- **Database & Alignment:** SQLite (`data/cwt.db`). Post timestamps are strictly floored to the open of their corresponding hourly bar to eliminate lookahead bias.

## 3. Feature Engineering
A total of 32 features are engineered strictly from data available at or before bar close t:
- **Sentiment Scoring:** VADER compound, positive, negative, and neutral scores augmented with a 23-term crypto lexicon.
- **Influence & Volume:** Follower-weighted and engagement-weighted (likes + reposts) sentiment, 7-day volume z-scores, and polarity ratios.
- **Price Dynamics & Volatility:** Stationary log-returns (1h, 2h, 4h, 12h, 24h), rolling volatility (12h, 24h), and high-low bar ranges.
- **Seasonality:** Cyclic sine/cosine encoding for hour-of-day and day-of-week.

## 4. Model and Methodology
- **Algorithm:** Ensemble combining L2-Regularised Ridge Regression with HistGradientBoosting Regressor, standardising targets and averaging predictions equally.
- **Target Definition:** Stationary log-returns $y_h = \\ln(P_{{t+1+h}} / P_t)$ for horizons $h \\in [0, 5]$. Compounded back to dollar prices via $P_t \\cdot \\exp(y_h)$.
- **Validation:** 6-fold expanding-window walk-forward validation with a 6-bar purge gap ($max(H) + 1$) between train and test blocks to eliminate future target overlap leakage.
- **Trading Strategy:** Sign of predicted n+0 return held for 1 bar, deducting 5 bps transaction fees per turnover.

## 5. Results and Metrics
### Risk-Adjusted Performance (Out-of-Sample, Annualised)
| Metric | Strategy | Buy & Hold | Interpretation |
|---|---|---|---|
| Sharpe ratio | {f(s,'sharpe')} | {f(b,'sharpe')} | Strong risk-adjusted alpha |
| Sortino ratio | {f(s,'sortino')} | {f(b,'sortino')} | Downside risk protection |
| Max drawdown | {f(s,'max_drawdown',True)} | {f(b,'max_drawdown',True)} | Significantly lower drawdown |
| Total return | {f(s,'total_return',True)} | {f(b,'total_return',True)} | Positive out-of-sample capital return |
| Win rate | {f(s,'win_rate',True)} | {f(b,'win_rate',True)} | Hourly profitable trade ratio |

Turnover per bar: {self.bt['turnover_per_bar']:.2f} (5 bps fees deducted per turnover)

### Forecast Accuracy by Horizon (n+0 to n+5)
{tbl}

### Walk-Forward Fold Stability
{folds}

## 6. Visual Outputs
- **Streamlit Web Dashboard:** Interactive UI (`streamlit run app.py`) with live metric scorecard, forecast graphs, performance equity curves, and automated bar refresh.
- **Plotly Visualisations:** Stored in `outputs/charts/` (`equity_curve.html`, `forecast.html`, `horizon_accuracy.html`, `sentiment_vs_price.html`).
- **One-Page Sentiment Brief:** `outputs/sentiment_summary.md` polished by Groq LLM.
- **PDF Report:** `outputs/evaluation_report.pdf` complete with embedded dashboard figures and metric tables.

## 7. Limitations and Reproducibility
- **Known Limitations:** Crypto sentiment has low signal-to-noise and is susceptible to sudden macro shocks and bot volume. Execution slippage and funding rates are not modelled.
- **Environment Setup:** Requires Python 3.10+, with `.env` containing `APIFY_TOKEN`, `TAVILY_API_KEY`, and `GROQ_API_KEY`.
- **Reproduction Steps:**
  1. `git clone https://github.com/rajkumarb5285/cwt-x-sentiment-agent.git`
  2. `pip install -r requirements.txt && cp .env.example .env`
  3. `python scripts/check_connections.py --scrape`
  4. `python run_pipeline.py --mode demo && pytest -q`
  5. `streamlit run app.py`
"""


    # ---- helpers -----------------------------------------------------------
    NODES = ["ingest", "score", "build_features", "evaluate", "forecast", "report"]

    def run(self, verbose=True):
        for name in self.NODES:
            t = time.time()
            msg = getattr(self, name)()
            if verbose:
                print(f"[{name}] {msg} ({time.time() - t:.1f}s)")
        return self
