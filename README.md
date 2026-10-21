# CWT X Sentiment Price Prediction Agent

> Predicts Bitcoin's price at **n+0 … n+5 hourly bars** using **X (Twitter) sentiment and price data only**, validated with **expanding-window walk-forward** and reported with **Sharpe, Sortino and Max Drawdown**.

```
Apify X scraper ─┐
Binance prices ──┼─► SQLite ─► sentiment (VADER/FinBERT) ─► features ─► walk-forward model ─► forecast
Tavily news ─────┘   (UTC-aligned)                          (sentiment + price only)         + backtest
                                                                                   │
                                                  charts · 1-page summary · evaluation_report.md · live dashboard
```

Orchestrated as a **LangGraph workflow** (`src/graph.py`): `ingest → score → build_features → evaluate → forecast → report`.

---

## Quick start

### Prerequisites
- Python 3.10+
- API keys: `APIFY_TOKEN`, `TAVILY_API_KEY` (see `.env.example`)

### Setup

**Linux / macOS:**
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add APIFY_TOKEN and TAVILY_API_KEY
```

**Windows (PowerShell):**
```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env   # add APIFY_TOKEN and TAVILY_API_KEY
```

### Run

```bash
# 1. Verify API connections (tiny credit use)
python scripts/check_connections.py --scrape

# 2. Demo run (offline, synthetic data — verifies the pipeline works)
python run_pipeline.py --mode demo

# 3. Real run: 180-day backfill (slow, uses Apify credits)
python run_pipeline.py --mode live

# 4. Load historical tweets from CSV (optional)
python run_pipeline.py --mode live --csv historical_tweets.csv

# 5. Launch the live dashboard
streamlit run app.py

# 6. Headless real-time loop (no dashboard)
python live.py --mode live

# 7. Run tests
pytest -q
```

---

## Architecture

| Layer | Choice | Rationale |
|---|---|---|
| **Language** | Python | ML ecosystem, as specified by the brief |
| **X data** | Apify (`apidojo/tweet-scraper`) | Required by brief. Returns text, timestamps, engagement, followers |
| **News** | Tavily | Required by brief. Used in summary only, never as a model feature |
| **Prices** | Binance public API + yfinance fallback | No key needed, hourly data, 24/7. Dual source for resilience |
| **Storage** | SQLite | No server setup, reviewer can run immediately. ~100k rows fits easily |
| **Sentiment** | VADER + crypto lexicon (FinBERT optional) | Fast, no GPU needed. FinBERT available via `config.yaml` |
| **Models** | Ridge + HistGradientBoosting ensemble | Stable linear baseline + non-linear, averaged to reduce overfitting |
| **Validation** | Expanding-window walk-forward, purge gap | Prevents future data leakage. Tests enforce this |
| **Metrics** | Sharpe, Sortino, Max Drawdown | As specified by brief, with fee-adjusted strategy returns |
| **Charts** | Plotly (interactive HTML) | Opens in any browser without installation |
| **Orchestration** | LangGraph | Explicit workflow graph, extensible |
| **Dashboard** | Streamlit | Real-time web view, auto-refreshes each bar |

---

## Design decisions

- **Timing / no look-ahead:** a row at bar-open `ts_i` uses only data available at that bar's *close*. Target `y_h = log(close[i+1+h] / close[i])`; n+0 is the bar currently forming. Tests in `tests/` enforce this.
- **Hourly bars, not daily:** 180 days × 24 = ~4,300 bars for training. Daily would give only 180.
- **Predict returns, convert to prices:** returns are stationary and more stable to model.
- **Validation:** expanding window, 6 folds, purge gap of `max(H)+1 = 6` bars so training targets never overlap the test block.
- **Features (sentiment + price only):** engagement/follower-weighted sentiment, post volume z-score, pos/neg ratios, sentiment lags/MAs/changes, lagged returns, volatility, bar range, hour/day-of-week. Tavily news is *not* a feature; it feeds the summary.
- **Models:** Ridge + HistGradientBoosting ensemble, one per horizon, predicting returns (converted to prices).
- **Backtest:** sign of predicted n+0 return, held one bar, 5 bps fee per turnover. Annualised with 365×24 bars/year.
- **Resilience:** Binance host fallback + yfinance; Apify chunked/resumable; each live cycle failure is logged and retried next bar.
- **Bitcoin as asset:** trades 24/7 (no weekend gaps), heavy X volume.

---

## Outputs (`outputs/`)

| File | Description |
|---|---|
| `evaluation_report.md` | Walk-forward Sharpe, Sortino, MaxDD vs buy-and-hold, accuracy per horizon (Markdown) |
| `evaluation_report.pdf` | Professional publication-quality evaluation report (PDF) |
| `sentiment_summary.md` | One-page sentiment outlook with crowd behaviour and model forecast |
| `forecast_latest.csv` | Latest n+0 to n+5 price predictions |
| `predictions_oos.csv` | All out-of-sample predictions from walk-forward validation |
| `features.csv` | Full feature matrix |
| `charts/*.html` | Interactive Plotly charts: sentiment vs price, equity curves, pred vs actual |

---

## Project structure

```
cwt-x-sentiment-agent/
├── run_pipeline.py          # CLI: full pipeline (demo or live)
├── live.py                  # Headless real-time loop
├── app.py                   # Streamlit dashboard
├── config.yaml              # All tuneable parameters
├── requirements.txt
├── .env.example
├── src/
│   ├── pipeline.py          # Pipeline class (6 LangGraph nodes)
│   ├── graph.py             # LangGraph workflow
│   ├── config.py            # Config loader
│   ├── db.py                # SQLite storage layer
│   ├── sentiment.py         # VADER/FinBERT scoring
│   ├── features.py          # Feature engineering
│   ├── model.py             # ML models + walk-forward
│   ├── backtest.py          # Strategy backtest
│   ├── viz.py               # Plotly charts
│   ├── summary.py           # One-page sentiment summary
│   └── ingest/
│       ├── apify_x.py       # X scraping via Apify
│       ├── prices.py        # Binance/yfinance price fetcher
│       ├── tavily_news.py   # Tavily news fetcher
│       └── demo.py          # Synthetic data generator
├── tests/
│   ├── test_pipeline.py     # Leakage + alignment tests
│   └── test_additional.py   # Sentiment, backtest, DB tests
├── scripts/
│   └── check_connections.py # API connection verifier
├── data/                    # SQLite databases (gitignored)
└── outputs/                 # Reports, charts, CSVs
```

---

## Submission Checklist (CrowdWisdomTrading)

To submit to `gilad@crowdwisdomtrading.com`:
1. **GitHub/GitLab repository link**: Push this repository and include the URL.
2. **APIFY / Tavily / Exa credentials & Actor IDs**:
   - Apify Actor ID: `apidojo/tweet-scraper` (in `config.yaml`).
   - Include your `APIFY_TOKEN`, `TAVILY_API_KEY` (or `EXA_API_KEY`) in your email.
3. **Evaluation report**: Attach `outputs/evaluation_report.pdf` (and/or `outputs/evaluation_report.md`), featuring out-of-sample Sharpe Ratio, Sortino Ratio, and Max Drawdown.

---

## Reproducibility & Service Support

- **Apify Actor**: `apidojo/tweet-scraper` (configured in `config.yaml`).
- **News Integration**: Dual provider support — queries **Tavily** or **Exa** based on `.env` keys (`TAVILY_API_KEY` or `EXA_API_KEY`).
- **Price Feeds**: Binance public API (`BTCUSDT`), with automatic failover to Yahoo Finance (`BTC-USD`).
- **Security**: Never commit `.env`. Tokens are read strictly from local environment variables.
