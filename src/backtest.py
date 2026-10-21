"""Turn OOS predictions into a trading signal and compute Sharpe / Sortino / Max Drawdown."""
import numpy as np
import pandas as pd
from src.config import bars_per_year


def metrics(r, bpy):
    r = pd.Series(r).dropna()
    if len(r) < 2 or r.std() == 0:
        return {}
    eq = np.exp(r.cumsum())
    dd = eq / eq.cummax() - 1
    downside = np.sqrt(np.mean(np.minimum(r, 0) ** 2))
    active = r[r != 0]
    return dict(sharpe=r.mean() / r.std() * np.sqrt(bpy),
                sortino=r.mean() / downside * np.sqrt(bpy) if downside > 0 else np.nan,
                max_drawdown=float(dd.min()), total_return=float(eq.iloc[-1] - 1),
                ann_return=float(r.mean() * bpy), win_rate=float((active > 0).mean()) if len(active) else np.nan,
                n_bars=int(len(r)))


def run_backtest(oos, cfg):
    """Signal at close of bar i from predicted n+0 return -> hold through bar i+1; fees on turnover."""
    bt, bpy = cfg["backtest"], bars_per_year(cfg)
    fee = bt["fee_bps"] / 1e4
    pred, r = oos["pred_y0"], oos["y0"]
    pos = np.sign(pred) * (pred.abs() > bt["threshold"])
    if bt["long_only"]:
        pos = pos.clip(lower=0)
    turnover = pos.diff().abs().fillna(pos.abs())
    pnl = pos * r - fee * turnover
    frame = pd.DataFrame({"pos": pos, "strategy": pnl, "buy_hold": r, "fold": oos["fold"]})
    frame["equity"] = np.exp(frame.strategy.cumsum())
    frame["bh_equity"] = np.exp(frame.buy_hold.cumsum())
    res = {"strategy": metrics(pnl, bpy), "buy_hold": metrics(r, bpy),
           "turnover_per_bar": float(turnover.mean()),
           "per_fold_sharpe": {int(f): metrics(g.strategy, bpy).get("sharpe") for f, g in frame.groupby("fold")}}
    return res, frame
