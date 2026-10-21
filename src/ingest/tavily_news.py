"""Recent, relevant news via Tavily or Exa (used for the written summary only, never as a model feature)."""
import os
import requests
import pandas as pd


def _fetch_exa(cfg, token=None):
    """Fetch recent news using Exa.ai REST API."""
    key = token or os.environ.get("EXA_API_KEY")
    if not key:
        return pd.DataFrame()
    url = "https://api.exa.ai/search"
    headers = {"x-api-key": key, "Content-Type": "application/json"}
    payload = {
        "query": f"{cfg['asset']['name']} cryptocurrency price market news",
        "numResults": cfg.get("tavily", {}).get("max_results", 8),
        "contents": {"text": {"maxCharacters": 600}}
    }
    resp = requests.post(url, headers=headers, json=payload, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    rows = []
    for r in data.get("results", []):
        pub = r.get("publishedDate")
        ts = pd.to_datetime(pub, utc=True, errors="coerce") if pub else pd.Timestamp.now(tz="UTC")
        rows.append(dict(
            url=r.get("url", ""),
            ts=ts,
            title=r.get("title", ""),
            content=(r.get("text") or "")[:600],
            score=float(r.get("score") or 0.0)
        ))
    df = pd.DataFrame(rows)
    if not df.empty:
        df["ts"] = df["ts"].fillna(pd.Timestamp.now(tz="UTC"))
    return df


def fetch_news(cfg, token=None):
    """Fetch news from Tavily (preferred) or Exa (fallback), based on available API keys."""
    tavily_key = token or os.environ.get("TAVILY_API_KEY")
    exa_key = os.environ.get("EXA_API_KEY")

    if tavily_key:
        try:
            from tavily import TavilyClient
            client = TavilyClient(api_key=tavily_key)
            res = client.search(
                query=f"{cfg['asset']['name']} price news",
                topic="news",
                days=cfg["tavily"]["days"],
                max_results=cfg["tavily"]["max_results"]
            )
            rows = []
            for r in res.get("results", []):
                pub = r.get("published_date")
                ts = pd.to_datetime(pub, utc=True, errors="coerce") if pub else pd.Timestamp.now(tz="UTC")
                rows.append(dict(
                    url=r["url"],
                    ts=ts,
                    title=r.get("title", ""),
                    content=(r.get("content") or "")[:600],
                    score=r.get("score", 0.0)
                ))
            df = pd.DataFrame(rows)
            if not df.empty:
                df["ts"] = df["ts"].fillna(pd.Timestamp.now(tz="UTC"))
                return df
        except Exception as e:
            print(f"[tavily] news search failed: {e}")
            if not exa_key:
                raise

    if exa_key:
        try:
            df = _fetch_exa(cfg, exa_key)
            if not df.empty:
                return df
        except Exception as e:
            print(f"[exa] news search failed: {e}")
            if not tavily_key:
                raise

    if not tavily_key and not exa_key:
        raise ValueError("Neither TAVILY_API_KEY nor EXA_API_KEY is configured in .env.")

    return pd.DataFrame(columns=["url", "ts", "title", "content", "score"])
