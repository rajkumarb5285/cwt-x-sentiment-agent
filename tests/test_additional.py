"""Additional tests: sentiment scoring, backtest metrics, edge cases, and database operations."""
import numpy as np
import pandas as pd
import pytest
from src.config import load_config
from src.ingest import demo
from src.features import build_features, aggregate_posts
from src.model import walk_forward, MultiHorizonModel, horizon_metrics
from src.backtest import metrics, run_backtest
from src.sentiment import clean, _score_vader
from src import db


def _data(days=40):
    cfg = load_config()
    cfg["horizons"] = [0, 1, 2, 3, 4, 5]
    posts, px, _ = demo.generate(cfg, days=days, seed=1)
    posts["sentiment"] = np.random.default_rng(0).uniform(-1, 1, len(posts))
    return cfg, posts, px


class TestSentiment:
    def test_clean_removes_urls_and_mentions(self):
        assert "http" not in clean("Check https://example.com @user bullish")
        assert "@" not in clean("@trader bitcoin is mooning")

    def test_vader_scores_bullish_text_positive(self):
        scores = _score_vader(["Bitcoin is bullish, rally incoming, mooning!"])
        assert scores[0] > 0

    def test_vader_scores_bearish_text_negative(self):
        scores = _score_vader(["BTC crash dump rekt bearish liquidated"])
        assert scores[0] < 0

    def test_vader_scores_neutral_text_near_zero(self):
        scores = _score_vader(["BTC trading sideways today"])
        assert abs(scores[0]) < 0.5

    def test_vader_batch_consistency(self):
        texts = ["bullish moon pump", "bearish crash dump", "neutral sideways"]
        scores = _score_vader(texts)
        assert len(scores) == 3
        assert scores[0] > scores[2] > scores[1]


class TestFeatures:
    def test_aggregate_posts_handles_empty(self):
        empty = pd.DataFrame(columns=["ts", "sentiment", "text", "likes", "reposts",
                                       "replies", "views", "followers"])
        result = aggregate_posts(empty, "1h")
        assert result.empty

    def test_build_features_rejects_empty_prices(self):
        cfg, posts, _ = _data()
        empty_px = pd.DataFrame(columns=["ts", "open", "high", "low", "close", "volume"])
        with pytest.raises(ValueError, match="No price data"):
            build_features(posts, empty_px, cfg)

    def test_feature_count_is_consistent(self):
        cfg, posts, px = _data()
        df, feats = build_features(posts, px, cfg)
        # All feature columns should exist in the DataFrame
        for f in feats:
            assert f in df.columns, f"Missing feature: {f}"
        # No target columns in features
        for h in cfg["horizons"]:
            assert f"y{h}" not in feats


class TestBacktest:
    def test_metrics_with_constant_returns(self):
        r = pd.Series([0.001] * 100)
        m = metrics(r, 8760)
        assert m["sharpe"] > 0
        assert m["max_drawdown"] >= -1e-10  # no drawdown with constant positive returns
        assert m["win_rate"] == 1.0

    def test_metrics_with_empty_returns(self):
        r = pd.Series([0.0])
        m = metrics(r, 8760)
        assert m == {}

    def test_run_backtest_produces_required_keys(self):
        cfg, posts, px = _data(60)
        cfg["model"].update(kind="ridge", n_splits=3, min_train_bars=300)
        df, feats = build_features(posts, px, cfg)
        oos = walk_forward(df, feats, cfg)
        bt, frame = run_backtest(oos, cfg)
        assert "strategy" in bt
        assert "buy_hold" in bt
        assert "turnover_per_bar" in bt
        assert "per_fold_sharpe" in bt
        assert "equity" in frame.columns
        assert "bh_equity" in frame.columns


class TestModel:
    def test_multi_horizon_model_predict_shape(self):
        cfg, posts, px = _data()
        df, feats = build_features(posts, px, cfg)
        H = cfg["horizons"]
        ycols = [f"y{h}" for h in H]
        lab = df.dropna(subset=feats + ycols)
        m = MultiHorizonModel("ridge", H).fit(lab[feats], lab[ycols])
        pred = m.predict(lab[feats].iloc[:5])
        assert pred.shape == (5, len(H))
        for h in H:
            assert f"y{h}" in pred.columns

    def test_horizon_metrics_columns(self):
        cfg, posts, px = _data(60)
        cfg["model"].update(kind="ridge", n_splits=3, min_train_bars=300)
        df, feats = build_features(posts, px, cfg)
        oos = walk_forward(df, feats, cfg)
        hm = horizon_metrics(oos, cfg["horizons"])
        expected_cols = {"horizon", "MAE_usd", "MAE_naive_usd", "skill_vs_naive",
                         "RMSE_ret", "dir_acc", "IC_spearman"}
        assert expected_cols.issubset(set(hm.columns))


class TestDatabase:
    def test_roundtrip_posts(self, tmp_path):
        conn = db.connect(str(tmp_path / "test.db"))
        posts = pd.DataFrame([{
            "id": "t1", "ts": pd.Timestamp("2024-01-01", tz="UTC"),
            "text": "bullish", "likes": 5, "reposts": 2, "replies": 1,
            "views": 100, "followers": 500, "author": "test"
        }])
        n = db.upsert_posts(conn, posts)
        assert n == 1
        result = db.read(conn, "posts")
        assert len(result) == 1
        assert result.iloc[0]["text"] == "bullish"

    def test_upsert_posts_idempotent(self, tmp_path):
        conn = db.connect(str(tmp_path / "test.db"))
        posts = pd.DataFrame([{
            "id": "t1", "ts": pd.Timestamp("2024-01-01", tz="UTC"),
            "text": "bullish", "likes": 5, "reposts": 2, "replies": 1,
            "views": 100, "followers": 500, "author": "test"
        }])
        db.upsert_posts(conn, posts)
        db.upsert_posts(conn, posts)
        assert db.count_posts(conn) == 1

    def test_read_rejects_invalid_table(self, tmp_path):
        conn = db.connect(str(tmp_path / "test.db"))
        with pytest.raises(ValueError, match="Invalid table name"):
            db.read(conn, "'; DROP TABLE posts; --")

    def test_upsert_empty_returns_zero(self, tmp_path):
        conn = db.connect(str(tmp_path / "test.db"))
        assert db.upsert_posts(conn, pd.DataFrame()) == 0
        assert db.upsert_posts(conn, None) == 0


class TestPdfReport:
    def test_generate_pdf_report_creates_valid_file(self, tmp_path):
        from src.pdf_report import generate_pdf_report
        cfg, posts, px = _data(40)
        cfg["model"].update(kind="ridge", n_splits=3, min_train_bars=200)
        df, feats = build_features(posts, px, cfg)
        oos = walk_forward(df, feats, cfg)
        hm = horizon_metrics(oos, cfg["horizons"])
        bt, _ = run_backtest(oos, cfg)
        out_pdf = str(tmp_path / "test_report.pdf")
        res = generate_pdf_report(cfg, df, bt, hm, out_pdf, synthetic=True)
        import os
        assert os.path.exists(res)
        assert os.path.getsize(res) > 1000
