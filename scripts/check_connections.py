"""Run this FIRST with your .env filled in. Makes one tiny call to each service."""
import os
import sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.config import load_config

cfg = load_config()
ok = True


def check(name, fn):
    global ok
    try:
        print(f"[ OK ] {name}: {fn()}")
    except Exception as e:
        ok = False
        print(f"[FAIL] {name}: {e}")


def binance():
    from src.ingest.prices import fetch_prices
    d = fetch_prices(cfg, pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=6))
    return f"{len(d)} bars, last close {d.close.iloc[-1]:,.0f}"


def apify():
    from apify_client import ApifyClient
    u = ApifyClient(os.environ["APIFY_TOKEN"]).user().get()
    # apify_client v3+ returns an object; older versions return a dict
    username = u.get("username") if isinstance(u, dict) else getattr(u, "username", str(u))
    return f"authenticated as {username}, actor = {cfg['apify']['actor_id']}"


def apify_scrape():
    from src.ingest.apify_x import fetch_range
    end = pd.Timestamp.now(tz="UTC").floor("D")
    cfg["apify"]["max_items_per_chunk"] = 20
    d = fetch_range(cfg, end - pd.Timedelta(days=1), end + pd.Timedelta(days=1))
    return f"{len(d)} posts; columns {list(d.columns)}; sample followers={d.followers.max() if len(d) else 'n/a'}"


def news():
    from src.ingest.tavily_news import fetch_news
    cfg["tavily"]["max_results"] = 3
    articles = fetch_news(cfg)
    provider = "Tavily" if os.environ.get("TAVILY_API_KEY") else ("Exa" if os.environ.get("EXA_API_KEY") else "None")
    return f"{len(articles)} articles fetched via {provider}"


check("Binance prices", binance)
check("Apify auth", apify)
if "--scrape" in sys.argv:
    check("Apify tiny scrape (uses a little credit)", apify_scrape)
check("News provider (Tavily/Exa)", news)
print("\nAll good. Next: python run_pipeline.py --mode live" if ok else "\nFix the failures above, then re-run.")
