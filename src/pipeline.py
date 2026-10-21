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
        data = ("**SYNTHETIC demo data**: numbers below validate the pipeline only, not market performance."
                if self.mode == "demo" else "Real data: Apify X posts + Binance prices.")
        return f"""# Evaluation report: X sentiment price prediction

{data}

## Setup
- Asset: {cfg['asset']['name']} ({cfg['asset']['symbol']}), bars: {cfg['bar']}, bars used: {len(self.df)}
- Inputs: X sentiment and price only (sentiment score, post volume, engagement-weighted score, lagged returns, volatility, time-of-day).
  Tavily news is used only in the written summary, never as a model feature.
- Targets: price at n+0 ... n+5, where n is the bar currently forming. Models predict log-return from the last closed bar.
- Model: {cfg['model']['kind']} (Ridge + HistGradientBoosting), one model per horizon.
- Validation: expanding-window walk-forward, {cfg['model']['n_splits']} folds, purge gap of {max(cfg['horizons']) + 1} bars between train and test. All metrics below are out-of-sample.
- Strategy: sign of predicted n+0 return, held one bar, {cfg['backtest']['fee_bps']} bps fee per unit turnover.

## Risk-adjusted performance (out-of-sample, annualised)
| Metric | Strategy | Buy & hold |
|---|---|---|
| Sharpe ratio | {f(s,'sharpe')} | {f(b,'sharpe')} |
| Sortino ratio | {f(s,'sortino')} | {f(b,'sortino')} |
| Max drawdown | {f(s,'max_drawdown',True)} | {f(b,'max_drawdown',True)} |
| Total return | {f(s,'total_return',True)} | {f(b,'total_return',True)} |
| Win rate | {f(s,'win_rate',True)} | {f(b,'win_rate',True)} |

Turnover per bar: {self.bt['turnover_per_bar']:.2f}

## Forecast accuracy by horizon
{tbl}

## Stability across folds
{folds}

## Limitations
- Sentiment-only inputs are a weak signal; crypto is noisy and results can vary by period.
- Scraped X data has sampling bias (search ranking, bots, deleted posts).
- Fees are simplified; slippage and funding costs are not modelled.
- Interactive charts are in `outputs/charts/`.
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
