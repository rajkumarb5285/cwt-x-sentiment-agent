"""Multi-horizon models + expanding-window walk-forward validation with a purge gap."""
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.base import clone
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _estimators(kind):
    ridge = make_pipeline(StandardScaler(), Ridge(alpha=50.0))
    gbm = HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=150,
                                        min_samples_leaf=40, l2_regularization=1.0, random_state=0)
    return {"ridge": [ridge], "gbm": [gbm]}.get(kind, [ridge, gbm])


class MultiHorizonModel:
    def __init__(self, kind, horizons):
        self.kind, self.H = kind, horizons

    def fit(self, X, Y):
        self.models, self.scale = {}, {}
        for h in self.H:
            y = Y[f"y{h}"].values
            self.scale[h] = y.std() + 1e-12
            self.models[h] = [clone(e).fit(X, y / self.scale[h]) for e in _estimators(self.kind)]
        return self

    def predict(self, X):
        out = {f"y{h}": np.mean([m.predict(X) for m in self.models[h]], axis=0) * self.scale[h] for h in self.H}
        return pd.DataFrame(out, index=X.index)


def walk_forward(df, feats, cfg):
    """Expanding-window OOS predictions. Train rows end `purge` bars before the test block, because
    a training row's targets look up to max(H)+1 bars ahead."""
    m, H = cfg["model"], cfg["horizons"]
    ycols = [f"y{h}" for h in H]
    lab = df.dropna(subset=feats + ycols)
    n, purge = len(lab), max(H) + 1
    min_train = min(m["min_train_bars"], int(n * 0.5))
    k = m["n_splits"]
    test_size = (n - min_train) // k
    out = []
    for i in range(k):
        a = min_train + i * test_size
        b = a + test_size if i < k - 1 else n
        tr, te = lab.iloc[:a - purge], lab.iloc[a:b]
        pr = MultiHorizonModel(m["kind"], H).fit(tr[feats], tr[ycols]).predict(te[feats])
        res = pd.DataFrame({"fold": i, "close": te.close}, index=te.index)
        for h in H:
            res[f"pred_y{h}"], res[f"y{h}"] = pr[f"y{h}"].values, te[f"y{h}"].values
        out.append(res)
    return pd.concat(out)


def horizon_metrics(oos, H):
    rows = []
    for h in H:
        y, p, c = oos[f"y{h}"], oos[f"pred_y{h}"], oos["close"]
        act, pred = c * np.exp(y), c * np.exp(p)
        mae, mae_naive = (pred - act).abs().mean(), (c - act).abs().mean()   # naive = "price stays at last close"
        rows.append(dict(horizon=f"n+{h}", MAE_usd=mae, MAE_naive_usd=mae_naive,
                         skill_vs_naive=1 - mae / mae_naive, RMSE_ret=float(np.sqrt(((y - p) ** 2).mean())),
                         dir_acc=float((np.sign(y) == np.sign(p)).mean()),
                         IC_spearman=float(spearmanr(y, p)[0])))
    return pd.DataFrame(rows)


def fit_final_and_forecast(df, feats, cfg):
    """Fit on all labelled history, forecast from the latest closed bar. Returns forecast DataFrame."""
    H = cfg["horizons"]
    ycols = [f"y{h}" for h in H]
    lab = df.dropna(subset=feats + ycols)
    model = MultiHorizonModel(cfg["model"]["kind"], H).fit(lab[feats], lab[ycols])
    last = df.iloc[[-1]]
    pr = model.predict(last[feats]).iloc[0]
    bar = pd.Timedelta(cfg["bar"])
    made_at = last.index[0] + bar          # open time of the bar currently forming = n+0
    base = float(last.close.iloc[0])
    return pd.DataFrame([dict(made_at=made_at, target_ts=made_at + h * bar, horizon=h,
                              pred_ret=float(pr[f"y{h}"]), pred_price=base * float(np.exp(pr[f"y{h}"])),
                              base_close=base) for h in H])
