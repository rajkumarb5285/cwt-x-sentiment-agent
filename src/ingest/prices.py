"""Price bars: Binance public klines (no key), yfinance fallback. Only CLOSED bars are returned."""
import time
import pandas as pd
import requests

HOSTS = ["https://api.binance.com", "https://data-api.binance.vision", "https://api.binance.us"]


def _closed_only(df, bar):
    now = pd.Timestamp.now(tz="UTC")
    return df[df["ts"] + pd.Timedelta(bar) <= now].reset_index(drop=True)


def fetch_binance(symbol, interval, start, end=None):
    start_ms = int(pd.Timestamp(start).timestamp() * 1000)
    end_ms = int((end or pd.Timestamp.now(tz="UTC")).timestamp() * 1000)
    last_err, rows = None, []
    for host in HOSTS:
        try:
            cur, rows = start_ms, []
            while cur < end_ms:
                r = requests.get(f"{host}/api/v3/klines", timeout=20, params=dict(
                    symbol=symbol, interval=interval, startTime=cur, endTime=end_ms, limit=1000))
                r.raise_for_status()
                data = r.json()
                if not data:
                    break
                rows += data
                cur = data[-1][0] + 1
                if len(data) < 1000:
                    break
                time.sleep(0.2)
            if rows:
                break
        except Exception as e:  # try next host
            last_err = e
    if not rows:
        raise RuntimeError(f"Binance fetch failed: {last_err}")
    df = pd.DataFrame(rows).iloc[:, :6]
    df.columns = ["ts", "open", "high", "low", "close", "volume"]
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df


def fetch_yfinance(yf_symbol, interval, start):
    import yfinance as yf
    d = yf.download(yf_symbol, start=start.tz_convert(None), interval=interval, progress=False, auto_adjust=False)
    d.columns = [c[0] if isinstance(c, tuple) else c for c in d.columns]
    d = d.reset_index().rename(columns={d.index.name or "Datetime": "ts", "Datetime": "ts", "Date": "ts",
                                         "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    d["ts"] = pd.to_datetime(d["ts"], utc=True)
    return d[["ts", "open", "high", "low", "close", "volume"]]


def fetch_prices(cfg, start, end=None):
    try:
        df = fetch_binance(cfg["asset"]["symbol"], cfg["bar"], start, end)
    except Exception as e:
        print(f"[prices] Binance failed ({e}); falling back to yfinance")
        df = fetch_yfinance(cfg["asset"]["yf_symbol"], cfg["bar"], start)
    return _closed_only(df, cfg["bar"])
