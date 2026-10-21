import numpy as np
import pandas as pd
from src.config import load_config
from src.ingest import demo
from src.features import build_features
from src.model import walk_forward


def _data(days=40):
    cfg = load_config()
    cfg["horizons"] = [0, 1, 2, 3, 4, 5]
    posts, px, _ = demo.generate(cfg, days=days, seed=1)
    posts["sentiment"] = np.random.default_rng(0).uniform(-1, 1, len(posts))
    return cfg, posts, px


def test_features_have_no_lookahead():
    """Features at bar i must be identical whether or not future bars/posts exist."""
    cfg, posts, px = _data()
    full, feats = build_features(posts, px, cfg)
    cut = px.ts.iloc[-200]
    part, _ = build_features(posts[posts.ts < cut + pd.Timedelta(cfg["bar"])], px[px.ts <= cut], cfg)
    common = part.index.intersection(full.index)
    pd.testing.assert_frame_equal(part.loc[common, feats], full.loc[common, feats], check_exact=False, atol=1e-9)


def test_target_alignment():
    cfg, posts, px = _data()
    df, _ = build_features(posts, px, cfg)
    i = 100
    t = df.index[i]
    for h in cfg["horizons"]:
        exp = np.log(df.close.iloc[i + 1 + h] / df.close.iloc[i])
        assert abs(df[f"y{h}"].iloc[i] - exp) < 1e-12, h


def test_walk_forward_is_strictly_out_of_sample():
    cfg, posts, px = _data(60)
    cfg["model"].update(kind="ridge", n_splits=3, min_train_bars=300)
    df, feats = build_features(posts, px, cfg)
    oos = walk_forward(df, feats, cfg)
    assert oos.index.is_unique and oos.index.is_monotonic_increasing
    assert oos.index.min() > df.index[299]
