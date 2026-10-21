"""Feature engineering from X sentiment + price ONLY.

Timing convention (no look-ahead):
  * ts = bar OPEN time (UTC). A row at ts_i is built from data available at the CLOSE of bar i
    (posts timestamped inside bar i, and bar i's OHLC).
  * Target y_h = log(close[i+1+h] / close[i]) for h = 0..5. So n+0 is the bar currently forming
    (the next one to close); predicted price = close[i] * exp(y_h).
"""
import numpy as np
import pandas as pd


def aggregate_posts(posts, bar):
    po = posts.dropna(subset=["sentiment"]).copy()
    if po.empty:
        return pd.DataFrame()
    po["bar"] = po["ts"].dt.floor(bar)
    po = po.drop_duplicates(["bar", "text"])                      # exact-text spam within a bar
    po["eng"] = po.likes + 2 * po.reposts + po.replies
    po["w"] = 1 + np.log1p(po.eng) + 0.5 * np.log1p(po.followers)   # engagement + reach weight
    po["ws"] = po.w * po.sentiment
    po["pos"] = (po.sentiment > 0.05).astype(float)
    po["neg"] = (po.sentiment < -0.05).astype(float)
    g = po.groupby("bar")
    return pd.DataFrame({
        "n_posts": g.size(), "sent_mean": g.sentiment.mean(), "sent_std": g.sentiment.std(),
        "sent_w": g.ws.sum() / g.w.sum(), "pos_ratio": g.pos.mean(), "neg_ratio": g.neg.mean(),
        "eng_log": np.log1p(g.eng.sum()), "views_log": np.log1p(g.views.sum())})


def build_features(posts, prices, cfg):
    bar, H = cfg["bar"], cfg["horizons"]
    if prices.empty:
        raise ValueError("No price data available. Run ingest first.")
    p = prices.sort_values("ts").drop_duplicates("ts").set_index("ts")[["open", "high", "low", "close"]]
    df = p.join(aggregate_posts(posts, bar), how="left")
    df["n_posts"] = df["n_posts"].fillna(0)
    for c in ("sent_mean", "sent_w", "pos_ratio", "neg_ratio"):
        df[c] = df[c].ffill(limit=3).fillna(0)
    for c in ("sent_std", "eng_log", "views_log"):
        df[c] = df[c].fillna(0)

    lc = np.log(df.close)
    df["ret_1"] = lc.diff()
    for k in range(1, 6):
        df[f"ret_l{k}"] = df.ret_1.shift(k)
    df["ret_6"], df["ret_24"] = lc.diff(6), lc.diff(24)
    df["vol_6"], df["vol_24"] = df.ret_1.rolling(6).std(), df.ret_1.rolling(24).std()
    df["range"] = (df.high - df.low) / df.close

    df["n_posts_log"] = np.log1p(df.n_posts)
    mu, sd = df.n_posts_log.rolling(72).mean(), df.n_posts_log.rolling(72).std()
    df["vol_z"] = (df.n_posts_log - mu) / (sd + 1e-6)
    for k in (1, 2, 3):
        df[f"sent_w_l{k}"] = df.sent_w.shift(k)
    df["sent_ma6"], df["sent_ma24"] = df.sent_w.rolling(6).mean(), df.sent_w.rolling(24).mean()
    df["sent_dev24"] = df.sent_w - df.sent_ma24
    df["sent_diff1"] = df.sent_w.diff()
    df["sent_x_vol"] = df.sent_w * df.vol_z

    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * df.index.hour / 24), np.cos(2 * np.pi * df.index.hour / 24)
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * df.index.dayofweek / 7), np.cos(2 * np.pi * df.index.dayofweek / 7)

    feats = [c for c in df.columns if c not in ("open", "high", "low", "close", "n_posts")]
    for h in H:
        df[f"y{h}"] = lc.shift(-(1 + h)) - lc
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feats)
    df.index.name = "ts"
    return df, feats
