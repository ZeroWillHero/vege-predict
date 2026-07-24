"""Train all 7 model families for all 6 vegetables and write the combined comparison table.

Usage:
    python scripts/train_all.py
"""

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

from src.models.catboost.train import train_all_vegetables as train_catboost
from src.models.hybrid_catboost_sarimax.train import train_all_vegetables as train_hybrid_catboost
from src.models.hybrid_xgboost_sarimax.train import train_all_vegetables as train_hybrid_xgboost
from src.models.lstm.train import train_all_vegetables as train_lstm
from src.models.random_forest.train import train_all_vegetables as train_random_forest
from src.models.sarimax.train import train_all_vegetables as train_sarimax
from src.models.xgboost.train import train_all_vegetables as train_xgboost
from src.utils.io import PROJECT_ROOT, load_config

MODEL_TRAINERS = {
    "sarimax": train_sarimax,
    "random_forest": train_random_forest,
    "xgboost": train_xgboost,
    "catboost": train_catboost,
    "hybrid_xgboost_sarimax": train_hybrid_xgboost,
    "hybrid_catboost_sarimax": train_hybrid_catboost,
    "lstm": train_lstm,
}


def main():
    config = load_config()
    all_results = []
    all_predictions = []

    for model_family, trainer in MODEL_TRAINERS.items():
        print(f"\n=== {model_family} ===")
        t0 = time.time()
        results, predictions = trainer(config)
        all_results.extend(results)
        all_predictions.extend(predictions)
        print(f"{model_family} done in {time.time() - t0:.1f}s")

    results_df = pd.DataFrame(all_results)
    metrics_path = PROJECT_ROOT / "results" / "metrics" / "all_results.csv"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(metrics_path, index=False)
    print(f"\nSaved {len(results_df)} rows to {metrics_path}")

    predictions_df = pd.concat(all_predictions, ignore_index=True)
    predictions_path = PROJECT_ROOT / "results" / "metrics" / "holdout_predictions.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"Saved {len(predictions_df)} holdout prediction rows to {predictions_path}")

    comparison = results_df.pivot(index="vegetable", columns="model", values="rmse")
    tables_path = PROJECT_ROOT / "results" / "tables" / "model_comparison.csv"
    tables_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(tables_path)
    print(f"Saved comparison table to {tables_path}")
    print("\nRMSE by vegetable x model (lower is better):")
    print(comparison.round(1).to_string())


if __name__ == "__main__":
    main()
