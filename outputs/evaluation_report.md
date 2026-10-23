# CWT Quantitative Evaluation Report: X Sentiment Agent

## 1. Project Overview
- **Project Name:** CWT X Sentiment Price Prediction Agent
- **Objective:** Quantify crowd psychology on X (Twitter) and evaluate whether social sentiment has statistically significant predictive power for short-term Bitcoin price direction across horizons n+0 to n+5.
- **Target Asset:** Bitcoin (BTCUSDT)
- **Forecasting Interval:** 1-hour discrete bars (predicting n+0 forming close through n+5 future hours).
- **Agent Architecture:** Autonomous LangGraph state graph (Ingest -> Score -> Build Features -> Evaluate -> Forecast -> Report) with scheduled auto-refresh and multi-modal reporting.

## 2. Data Collection
- **Coverage:** 4,287 total hourly bars (~178.6 days of data).
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
- **Target Definition:** Stationary log-returns $y_h = \ln(P_{t+1+h} / P_t)$ for horizons $h \in [0, 5]$. Compounded back to dollar prices via $P_t \cdot \exp(y_h)$.
- **Validation:** 6-fold expanding-window walk-forward validation with a 6-bar purge gap ($max(H) + 1$) between train and test blocks to eliminate future target overlap leakage.
- **Trading Strategy:** Sign of predicted n+0 return held for 1 bar, deducting 5 bps transaction fees per turnover.

## 5. Results and Metrics
### Risk-Adjusted Performance (Out-of-Sample, Annualised)
| Metric | Strategy | Buy & Hold | Interpretation |
|---|---|---|---|
| Sharpe ratio | 2.63 | 1.51 | Strong risk-adjusted alpha |
| Sortino ratio | 19.12 | 10.55 | Downside risk protection |
| Max drawdown | -16.6% | -36.8% | Significantly lower drawdown |
| Total return | 418.2% | 158.3% | Positive out-of-sample capital return |
| Win rate | 52.1% | 49.2% | Hourly profitable trade ratio |

Turnover per bar: 0.57 (5 bps fees deducted per turnover)

### Forecast Accuracy by Horizon (n+0 to n+5)
| Horizon | MAE ($) | Naive MAE ($) | Skill vs naive | Dir. acc | IC |
|---|---|---|---|---|---|
| n+0 | 111.9 | 110.7 | -1.02% | 54.9% | +0.125 |
| n+1 | 172.4 | 172.6 | +0.10% | 55.4% | +0.157 |
| n+2 | 224.3 | 224.5 | +0.10% | 56.5% | +0.188 |
| n+3 | 269.0 | 272.5 | +1.29% | 58.4% | +0.234 |
| n+4 | 308.6 | 316.9 | +2.60% | 59.9% | +0.269 |
| n+5 | 351.5 | 362.2 | +2.95% | 59.7% | +0.285 |

### Walk-Forward Fold Stability
- Fold 0: Sharpe -5.68
- Fold 1: Sharpe 1.13
- Fold 2: Sharpe 4.34
- Fold 3: Sharpe 11.09
- Fold 4: Sharpe 4.10
- Fold 5: Sharpe 5.29

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
