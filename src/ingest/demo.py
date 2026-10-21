"""SYNTHETIC data so the whole pipeline can run offline. Not real market data.
A latent slow 'crowd mood' drives both tweet sentiment and (weakly) the next-bar return."""
import numpy as np
import pandas as pd

BULL = ["$BTC looks very bullish, breakout incoming", "Bitcoin rally is strong, great momentum, buying more",
        "BTC holding support, love this setup, moon soon", "Huge accumulation by whales, bullish on bitcoin"]
BEAR = ["$BTC looks weak, expecting a dump", "Bitcoin crash incoming, sell now, terrible price action",
        "BTC rejected again, bearish and fearful, liquidations everywhere", "FUD everywhere, bitcoin could collapse"]
NEUT = ["BTC trading sideways today", "Watching bitcoin price, waiting for a move", "$BTC update for the day"]


def generate(cfg, days=180, seed=7):
    rng = np.random.default_rng(seed)
    bar = pd.Timedelta(cfg["bar"])
    n = int(pd.Timedelta(days=days) / bar)
    end = pd.Timestamp.now(tz="UTC").floor(cfg["bar"])
    idx = pd.date_range(end=end - bar, periods=n, freq=bar)
    alpha, h, ret = np.zeros(n), np.zeros(n), np.zeros(n)
    for t in range(1, n):
        alpha[t] = 0.97 * alpha[t - 1] + 0.25 * rng.normal()
        h[t] = 0.95 * h[t - 1] + 0.2 * rng.normal()
        ret[t] = 0.0005 * alpha[t - 1] + 0.004 * np.exp(0.3 * h[t]) * rng.normal()
    close = 40000 * np.exp(np.cumsum(ret))
    op = np.r_[close[0], close[:-1]]
    hi = np.maximum(op, close) * (1 + np.abs(rng.normal(0, 0.0008, n)))
    lo = np.minimum(op, close) * (1 - np.abs(rng.normal(0, 0.0008, n)))
    prices = pd.DataFrame(dict(ts=idx, open=op, high=hi, low=lo, close=close, volume=rng.lognormal(6, .4, n)))

    rows, k = [], 0
    for t in range(n):
        lam = 10 * np.exp(0.4 * abs(ret[t]) / 0.005)
        for _ in range(rng.poisson(lam)):
            mood = alpha[t] + 60 * ret[t] + rng.normal(0, 0.8)
            tpl = BULL if mood > 0.4 else BEAR if mood < -0.4 else NEUT
            ts = idx[t] + pd.Timedelta(seconds=float(rng.uniform(0, bar.total_seconds() - 1)))
            rows.append(dict(id=f"demo_{k}", ts=ts, text=f"{tpl[rng.integers(len(tpl))]} {rng.integers(10000, 99999)}",
                             likes=int(rng.lognormal(1.5, 1.5)), reposts=int(rng.lognormal(.8, 1.3)),
                             replies=int(rng.lognormal(.5, 1.0)), views=int(rng.lognormal(6, 1.5)),
                             followers=int(rng.lognormal(7, 2)), author=f"user{rng.integers(5000)}"))
            k += 1
    posts = pd.DataFrame(rows)
    news = pd.DataFrame([dict(url=f"https://example.com/demo-{i}", ts=end - pd.Timedelta(hours=3 * i),
                              title=t, content="Synthetic demo headline.", score=0.5)
                         for i, t in enumerate(["[DEMO] Bitcoin ETF flows stay positive",
                                                "[DEMO] Analysts watch key support after volatility",
                                                "[DEMO] Exchange reserves fall to multi-month low"])])
    return posts, prices, news
