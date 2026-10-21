"""SQLite store: posts, prices, news, forecasts. All timestamps are UTC ISO strings."""
import os
import sqlite3
import pandas as pd

SCHEMA = """
CREATE TABLE IF NOT EXISTS posts(id TEXT PRIMARY KEY, ts TEXT, text TEXT, likes INT, reposts INT,
  replies INT, views INT, followers INT, author TEXT, sentiment REAL);
CREATE INDEX IF NOT EXISTS idx_posts_ts ON posts(ts);
CREATE TABLE IF NOT EXISTS prices(ts TEXT PRIMARY KEY, open REAL, high REAL, low REAL, close REAL, volume REAL);
CREATE TABLE IF NOT EXISTS news(url TEXT PRIMARY KEY, ts TEXT, title TEXT, content TEXT, score REAL);
CREATE TABLE IF NOT EXISTS forecasts(made_at TEXT, target_ts TEXT, horizon INT, pred_price REAL,
  pred_ret REAL, base_close REAL, PRIMARY KEY(made_at, horizon));
"""
FMT = "%Y-%m-%dT%H:%M:%SZ"


def connect(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.executescript(SCHEMA)
    return conn


def _iso(s):
    return pd.to_datetime(s, utc=True).dt.strftime(FMT)


def upsert_posts(conn, df):
    if df is None or df.empty:
        return 0
    d = df.copy()
    d["ts"] = _iso(d["ts"])
    d["sentiment"] = None
    rows = d[["id", "ts", "text", "likes", "reposts", "replies", "views", "followers", "author", "sentiment"]]
    conn.executemany("INSERT OR IGNORE INTO posts VALUES (?,?,?,?,?,?,?,?,?,?)", rows.values.tolist())
    conn.commit()
    return len(rows)


def upsert_prices(conn, df):
    d = df.copy()
    d["ts"] = _iso(d["ts"])
    conn.executemany("INSERT OR REPLACE INTO prices VALUES (?,?,?,?,?,?)",
                     d[["ts", "open", "high", "low", "close", "volume"]].values.tolist())
    conn.commit()


def upsert_news(conn, df):
    if df is None or df.empty:
        return
    d = df.copy()
    d["ts"] = _iso(d["ts"])
    conn.executemany("INSERT OR REPLACE INTO news VALUES (?,?,?,?,?)",
                     d[["url", "ts", "title", "content", "score"]].values.tolist())
    conn.commit()


_VALID_TABLES = {"posts", "prices", "news", "forecasts"}
_VALID_ORDER = {"ts", "made_at", "target_ts", "url", "id"}


def read(conn, table, order="ts"):
    if table not in _VALID_TABLES:
        raise ValueError(f"Invalid table name: {table!r}")
    if order not in _VALID_ORDER:
        raise ValueError(f"Invalid order column: {order!r}")
    df = pd.read_sql(f"SELECT * FROM {table} ORDER BY {order}", conn)
    for c in ("ts", "made_at", "target_ts"):
        if c in df:
            df[c] = pd.to_datetime(df[c], utc=True)
    return df


def count_posts(conn, start=None, end=None):
    q, args = "SELECT COUNT(*) FROM posts", []
    if start is not None:
        q += " WHERE ts>=? AND ts<?"
        args = [start.strftime(FMT), end.strftime(FMT)]
    return conn.execute(q, args).fetchone()[0]


def save_forecast(conn, fc):
    d = fc.copy()
    for c in ("made_at", "target_ts"):
        d[c] = d[c].dt.strftime(FMT)
    conn.executemany("INSERT OR REPLACE INTO forecasts VALUES (?,?,?,?,?,?)",
                     d[["made_at", "target_ts", "horizon", "pred_price", "pred_ret", "base_close"]].values.tolist())
    conn.commit()
