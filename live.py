"""Real-time loop: every bar, refresh data -> rebuild features -> re-forecast n+0..n+5 -> refresh charts/summary.
Walk-forward evaluation is re-run once per 24h (cached), forecasts every cycle.
  python live.py --mode live            # runs forever
  python live.py --mode live --once     # single cycle
"""
import argparse
import time
import pandas as pd
from src.config import load_config
from src.pipeline import Pipeline


def cycle(cfg, mode):
    Pipeline(cfg, mode, full_eval=False).run()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["demo", "live"], default="live")
    ap.add_argument("--once", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    bar = pd.Timedelta(cfg["bar"])
    while True:
        try:
            cycle(cfg, a.mode)
        except Exception as e:
            print(f"[live] cycle failed, will retry next bar: {e}")
        if a.once:
            break
        now = pd.Timestamp.now(tz="UTC")
        nxt = now.floor(cfg["bar"]) + bar + pd.Timedelta(minutes=1)   # 1 min after the bar closes
        wait = max(30, (nxt - now).total_seconds())
        print(f"[live] sleeping {wait / 60:.1f} min until {nxt:%H:%M} UTC")
        time.sleep(wait)
