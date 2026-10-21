"""Full run: ingest -> sentiment -> features -> walk-forward -> forecast -> charts + reports.
  python run_pipeline.py --mode demo                 # offline, synthetic data (checks everything works)
  python run_pipeline.py --mode live                 # real Apify + Binance + Tavily (needs .env keys)
  python run_pipeline.py --mode live --csv hist.csv  # also load the provided historical tweets
"""
import argparse
from src.config import load_config
from src.graph import run_graph
from src.pipeline import Pipeline

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["demo", "live"], default="demo")
    ap.add_argument("--days", type=int)
    ap.add_argument("--csv", help="historical tweets CSV to load")
    ap.add_argument("--no-resume", action="store_true")
    a = ap.parse_args()
    run_graph(Pipeline(load_config(), a.mode, a.days, a.csv, resume=not a.no_resume))
    print("\nDone. Open outputs/ or run: streamlit run app.py")
