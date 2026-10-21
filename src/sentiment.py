"""Post-level sentiment in [-1, 1]. VADER + crypto lexicon (default) or FinBERT (optional)."""
import re
import pandas as pd

CRYPTO_LEX = {"bullish": 2.6, "bearish": -2.6, "moon": 2.4, "mooning": 2.6, "pump": 1.6, "dump": -2.2, "rekt": -2.6,
              "hodl": 1.0, "rally": 2.0, "crash": -3.0, "breakout": 1.8, "fud": -1.6, "ath": 2.0, "liquidated": -2.6,
              "liquidations": -2.2, "rug": -3.0, "accumulation": 1.5, "collapse": -2.8, "rejected": -1.5,
              "support": 0.8, "resistance": -0.5, "capitulation": -2.5, "surge": 2.0, "plunge": -2.6}
_vader = None
_finbert = None


def clean(text):
    text = re.sub(r"https?://\S+", " ", text)
    return re.sub(r"@\w+", " ", text).strip()


def _score_vader(texts):
    global _vader
    if _vader is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _vader = SentimentIntensityAnalyzer()
        _vader.lexicon.update(CRYPTO_LEX)
    return [_vader.polarity_scores(clean(t))["compound"] for t in texts]


def _score_finbert(texts, batch=64):
    global _finbert
    if _finbert is None:
        from transformers import pipeline
        _finbert = pipeline("text-classification", model="ProsusAI/finbert", top_k=None, truncation=True, max_length=128)
    out = []
    for i in range(0, len(texts), batch):
        for res in _finbert([clean(t) for t in texts[i:i + batch]]):
            d = {r["label"].lower(): r["score"] for r in res}
            out.append(d.get("positive", 0) - d.get("negative", 0))
    return out


def score_unscored(conn, cfg):
    df = pd.read_sql("SELECT id, text FROM posts WHERE sentiment IS NULL", conn)
    if df.empty:
        return 0
    fn = _score_finbert if cfg["sentiment"]["engine"] == "finbert" else _score_vader
    scores = fn(df["text"].tolist())
    conn.executemany("UPDATE posts SET sentiment=? WHERE id=?", list(zip(scores, df["id"])))
    conn.commit()
    return len(df)
