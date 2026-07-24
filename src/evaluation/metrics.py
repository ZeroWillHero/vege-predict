"""Shared evaluation metrics (MAE, RMSE, MAPE) and cross-model comparison tables."""

import numpy as np


def mae(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mape(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100)


def r2(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def evaluate(y_true, y_pred) -> dict:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def time_series_splits(n: int, n_splits: int, holdout: int):
    """Walk-forward CV split indices over the first (n - holdout) rows, plus the final holdout.

    Returns (cv_folds, holdout_slice) where cv_folds is a list of (train_idx, val_idx)
    numpy index arrays covering an expanding training window, and holdout_slice is the
    (train_idx, test_idx) pair for the final untouched evaluation window.
    """
    trainable = n - holdout
    fold_size = trainable // (n_splits + 1)
    cv_folds = []
    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        val_end = fold_size * (i + 1)
        cv_folds.append((np.arange(0, train_end), np.arange(train_end, val_end)))
    holdout_slice = (np.arange(0, trainable), np.arange(trainable, n))
    return cv_folds, holdout_slice
