# Evaluation report: X sentiment price prediction

Real data: Apify X posts + Binance prices.

## Setup
- Asset: Bitcoin (BTCUSDT), bars: 1h, bars used: 4263
- Inputs: X sentiment and price only (sentiment score, post volume, engagement-weighted score, lagged returns, volatility, time-of-day).
  Tavily news is used only in the written summary, never as a model feature.
- Targets: price at n+0 ... n+5, where n is the bar currently forming. Models predict log-return from the last closed bar.
- Model: ensemble (Ridge + HistGradientBoosting), one model per horizon.
- Validation: expanding-window walk-forward, 6 folds, purge gap of 6 bars between train and test. All metrics below are out-of-sample.
- Strategy: sign of predicted n+0 return, held one bar, 5 bps fee per unit turnover.

## Risk-adjusted performance (out-of-sample, annualised)
| Metric | Strategy | Buy & hold |
|---|---|---|
| Sharpe ratio | 4.03 | -0.84 |
| Sortino ratio | 5.89 | -1.18 |
| Max drawdown | -15.7% | -36.8% |
| Total return | 65.8% | -9.9% |
| Win rate | 51.9% | 49.2% |

Turnover per bar: 0.59

## Forecast accuracy by horizon
| Horizon | MAE ($) | Naive MAE ($) | Skill vs naive | Dir. acc | IC |
|---|---|---|---|---|---|
| n+0 | 88.5 | 89.3 | +0.94% | 54.8% | +0.133 |
| n+1 | 128.6 | 130.2 | +1.25% | 56.4% | +0.164 |
| n+2 | 158.8 | 161.5 | +1.69% | 57.1% | +0.199 |
| n+3 | 183.2 | 188.8 | +2.96% | 58.6% | +0.241 |
| n+4 | 203.7 | 211.9 | +3.89% | 59.5% | +0.264 |
| n+5 | 225.0 | 235.9 | +4.62% | 60.0% | +0.282 |

## Stability across folds
- Fold 0: Sharpe -6.17
- Fold 1: Sharpe 2.95
- Fold 2: Sharpe 0.43
- Fold 3: Sharpe 8.94
- Fold 4: Sharpe 2.90
- Fold 5: Sharpe 15.84

## Limitations
- Sentiment-only inputs are a weak signal; crypto is noisy and results can vary by period.
- Scraped X data has sampling bias (search ranking, bots, deleted posts).
- Fees are simplified; slippage and funding costs are not modelled.
- Interactive charts are in `outputs/charts/`.
