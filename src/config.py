import os
from pathlib import Path
import pandas as pd
import yaml
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def load_config(path=None):
    cfg = yaml.safe_load(Path(path or ROOT / "config.yaml").read_text())
    cfg["root"] = str(ROOT)
    for k in ("db", "outputs"):
        cfg["paths"][k] = str(ROOT / cfg["paths"][k])
    return cfg


def bar_td(cfg):
    return pd.Timedelta(cfg["bar"])


def bars_per_year(cfg):
    return pd.Timedelta(days=365) / bar_td(cfg)


def has_live_keys():
    return bool(os.getenv("APIFY_TOKEN")) and bool(os.getenv("TAVILY_API_KEY"))
