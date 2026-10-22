"""X (Twitter) scraping through Apify actors + CSV loader for the provided historical tweets."""
import hashlib
import os
import pandas as pd
from src import db


def _first(d, *keys, default=None):
    for k in keys:
        if d.get(k) is not None:
            return d[k]
    return default


def _int(x):
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return 0


def normalise(items):
    """Map heterogeneous actor / CSV records into our posts schema."""
    rows = []
    for it in items:
        if it.get("noResults") or it.get("isRetweet"):
            continue
        text = _first(it, "text", "fullText", "full_text", "content")
        ts = _first(it, "createdAt", "created_at", "timestamp", "date")
        if not text or not ts:
            continue
        a = it.get("author") or it.get("user") or {}
        a = a if isinstance(a, dict) else {}
        pid = _first(it, "id", "id_str", "tweetId", "tweet_id")
        if pid is None:
            pid = hashlib.md5(f"{text}{ts}".encode()).hexdigest()
        rows.append(dict(
            id=str(pid), ts=pd.to_datetime(ts, utc=True, errors="coerce"), text=str(text),
            likes=_int(_first(it, "likeCount", "favorite_count", "likes", "like_count")),
            reposts=_int(_first(it, "retweetCount", "retweet_count", "reposts", "repost_count")),
            replies=_int(_first(it, "replyCount", "reply_count", "replies")),
            views=_int(_first(it, "viewCount", "views", "view_count")),
            followers=_int(_first(a, "followers", "followersCount", "followers_count")
                           or _first(it, "followers", "author_followers")),
            author=str(_first(a, "userName", "screen_name", "username") or _first(it, "author", "username") or ""),
        ))
    df = pd.DataFrame(rows)
    return df.dropna(subset=["ts"]) if not df.empty else df


def fetch_range(cfg, start, end, token=None):
    """One Apify actor run covering [start, end) (day granularity, X advanced-search operators)."""
    from apify_client import ApifyClient
    client = ApifyClient(token or os.environ["APIFY_TOKEN"])
    s, e = start.strftime("%Y-%m-%d"), (end + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    terms = [f"{q} lang:en since:{s} until:{e}" for q in cfg["asset"]["queries"]]
    run = client.actor(cfg["apify"]["actor_id"]).call(run_input={
        "searchTerms": terms, "maxItems": cfg["apify"]["max_items_per_chunk"],
        "sort": "Latest", "tweetLanguage": "en"})
    # apify_client v3+ returns a Run object; older versions return a dict
    dataset_id = run["defaultDatasetId"] if isinstance(run, dict) else getattr(run, "default_dataset_id", None) or run.get("defaultDatasetId")
    items = list(client.dataset(dataset_id).iterate_items())
    df = normalise(items)
    if not df.empty:
        df = df[(df.ts >= start) & (df.ts < end)]
    return df


def backfill(cfg, conn, days, resume=True):
    end = pd.Timestamp.now(tz="UTC").floor("D") + pd.Timedelta(days=1)
    cur, step = end - pd.Timedelta(days=days), pd.Timedelta(days=cfg["apify"]["chunk_days"])
    total = 0
    while cur < end:
        nxt = min(cur + step, end)
        if resume and db.count_posts(conn, cur, nxt) >= cfg["apify"]["min_posts_to_skip_chunk"]:
            print(f"[apify] {cur:%Y-%m-%d} -> {nxt:%Y-%m-%d}: already have data, skipping")
        else:
            try:
                df = fetch_range(cfg, cur, nxt)
                total += db.upsert_posts(conn, df)
                print(f"[apify] {cur:%Y-%m-%d} -> {nxt:%Y-%m-%d}: {len(df)} posts")
            except Exception as ex:
                print(f"[apify] chunk {cur:%Y-%m-%d} failed: {ex}")
        cur = nxt
    return total


def load_csv(conn, path):
    """Load the provided historical tweets (any CSV with text + created-at-like columns)."""
    raw = pd.read_csv(path)
    df = normalise(raw.to_dict("records"))
    n = db.upsert_posts(conn, df)
    print(f"[csv] loaded {n} historical posts from {path}")
    return n
