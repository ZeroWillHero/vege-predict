"""Shared CLI/evaluation plumbing reused by every model's train.py (not a model itself)."""

import argparse

import numpy as np
import pandas as pd

from src.evaluation.metrics import evaluate, interval_coverage, prediction_interval, time_series_splits
from src.features.feature_engineering import build_features, get_feature_columns
from src.utils.io import get_processed_path, save_model


def parse_vegetable_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vegetable", help="Single vegetable; omit to run all configured vegetables")
    return parser.parse_args()


def load_features(vegetable: str, config: dict):
    df = pd.read_csv(get_processed_path(vegetable, config), parse_dates=["date"])
    feat = build_features(df, config)
    feature_cols = get_feature_columns(feat)
    return feat, feature_cols


def run_training(vegetable: str, model_family: str, config: dict, fit_predict, fit_final, keras: bool = False):
    """Walk-forward CV + final holdout evaluation + production refit, for any model family.

    fit_predict(train_df, test_df, feature_cols) -> array of predictions aligned to test_df.
    fit_final(full_df, feature_cols, vegetable) -> fitted model object, saved as the production artifact.

    Returns (result, predictions_df): result is the metrics dict; predictions_df holds the
    holdout-period date/actual/predicted series, plus a predicted_lower/predicted_upper
    prediction interval (used for forecast-vs-actual plots and the API's forecast rows).
    """
    feat, feature_cols = load_features(vegetable, config)
    n = len(feat)
    cv_folds, (holdout_train_idx, holdout_test_idx) = time_series_splits(
        n, config["forecasting"]["cv_splits"], config["forecasting"]["holdout_weeks"]
    )

    cv_metrics = []
    cv_residuals = []
    for train_idx, val_idx in cv_folds:
        train_df, val_df = feat.iloc[train_idx], feat.iloc[val_idx]
        preds = fit_predict(train_df, val_df, feature_cols)
        cv_metrics.append(evaluate(val_df["target"].values, preds))
        cv_residuals.append(val_df["target"].values - preds)
    cv_avg = {k: sum(m[k] for m in cv_metrics) / len(cv_metrics) for k in cv_metrics[0]} if cv_metrics else {}

    holdout_train_df = feat.iloc[holdout_train_idx]
    holdout_test_df = feat.iloc[holdout_test_idx]
    holdout_preds = fit_predict(holdout_train_df, holdout_test_df, feature_cols)
    holdout_metrics = evaluate(holdout_test_df["target"].values, holdout_preds)

    # Prediction interval: calibrated from the walk-forward CV folds' out-of-sample residuals
    # (not the holdout itself), then applied to the holdout predictions — so the holdout also
    # serves as a genuine out-of-sample check of how well the interval is calibrated.
    confidence = config["forecasting"].get("interval_confidence", 0.8)
    lower_offset, upper_offset = prediction_interval(np.concatenate(cv_residuals), confidence)
    predicted_lower = holdout_preds + lower_offset
    predicted_upper = holdout_preds + upper_offset
    coverage = interval_coverage(holdout_test_df["target"].values, predicted_lower, predicted_upper)

    predictions_df = pd.DataFrame(
        {
            "vegetable": vegetable,
            "model": model_family,
            "date": holdout_test_df["date"].values,
            "actual": holdout_test_df["target"].values,
            "predicted": holdout_preds,
            "predicted_lower": predicted_lower,
            "predicted_upper": predicted_upper,
        }
    )

    final_model = fit_final(feat, feature_cols, vegetable)
    save_model(final_model, model_family, vegetable, config, keras=keras)

    result = {"vegetable": vegetable, "model": model_family, **holdout_metrics}
    result.update({f"cv_{k}": v for k, v in cv_avg.items()})
    result.update({"interval_confidence": confidence, "interval_coverage": coverage})
    print(
        f"{vegetable}/{model_family}: holdout RMSE={holdout_metrics['rmse']:.2f} "
        f"MAPE={holdout_metrics['mape']:.2f}% R2={holdout_metrics['r2']:.3f} "
        f"interval_coverage={coverage:.2f} (nominal {confidence:.2f})"
    )
    return result, predictions_df


def run_for_vegetables(vegetables, model_family: str, config: dict, fit_predict, fit_final, keras: bool = False):
    results, predictions = [], []
    for v in vegetables:
        result, preds = run_training(v, model_family, config, fit_predict, fit_final, keras=keras)
        results.append(result)
        predictions.append(preds)
    return results, predictions
