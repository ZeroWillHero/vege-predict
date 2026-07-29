"""Generate genuine future-week forecasts for every model family, for every vegetable (or one,
via --vegetable), and write results/metrics/future_predictions.csv. Requires trained_models/
artifacts to already exist (run scripts/train_all.py first). See src/inference/future_forecast.py
for the recursive, per-family forecasting design.

Usage:
    python scripts/predict_future.py
    python scripts/predict_future.py --vegetable carrot
    python scripts/predict_future.py --weeks 6
"""

import argparse
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.inference.future_forecast import recursive_forecast
from src.utils.io import PROJECT_ROOT, load_config

MODEL_FAMILIES = [
    "sarimax",
    "random_forest",
    "xgboost",
    "catboost",
    "hybrid_xgboost_sarimax",
    "hybrid_catboost_sarimax",
    "hybrid_random_forest_sarimax",
    "hybrid_lstm_sarimax",
    "lstm",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vegetable", help="Single vegetable; omit to run all configured vegetables")
    parser.add_argument("--weeks", type=int, help="Forecast horizon in weeks; overrides config default")
    args = parser.parse_args()

    config = load_config()
    vegetables = [args.vegetable] if args.vegetable else config["vegetables"]

    frames = []
    for vegetable in vegetables:
        for model_family in MODEL_FAMILIES:
            print(f"{vegetable}/{model_family}...", end=" ", flush=True)
            df = recursive_forecast(vegetable, model_family, config, horizon_weeks=args.weeks)
            frames.append(df)
            print(f"{len(df)} weeks, predicted range [{df['predicted'].min():.1f}, {df['predicted'].max():.1f}]")

    result = pd.concat(frames, ignore_index=True)
    out_path = PROJECT_ROOT / "results" / "metrics" / "future_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out_path, index=False)
    print(f"\nSaved {len(result)} rows to {out_path}")


if __name__ == "__main__":
    main()
