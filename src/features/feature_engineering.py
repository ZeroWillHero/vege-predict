"""Lag, rolling, and calendar feature construction shared across all models."""

import pandas as pd

NON_FEATURE_COLUMNS = ["date", "target", "retail_price", "wholesale_price"]

LAG_WEEKS = [1, 2, 4, 8]
ROLLING_WINDOWS = [4, 8, 12]
EXOG_COLUMNS = ["average_temperature", "average_rainfall", "diesel_price"]


def build_features(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    """Turn a processed per-vegetable frame into a model-ready feature frame.

    Adds price lags/rolling stats (computed strictly from past values), calendar
    features, a Maha/Yala season flag, and lagged exogenous features, alongside the
    contemporaneous exogenous columns already in `df`. Rows with NaN warm-up values
    (from the longest lag/rolling window) are dropped.
    """
    target_col = config["forecasting"]["target_column"]
    df = df.sort_values(by="date").reset_index(drop=True).copy()
    price = df[target_col]

    for lag in LAG_WEEKS:
        df[f"price_lag_{lag}"] = price.shift(lag)

    shifted = price.shift(1)
    for window in ROLLING_WINDOWS:
        df[f"price_rolling_mean_{window}"] = shifted.rolling(window).mean()
        df[f"price_rolling_std_{window}"] = shifted.rolling(window).std()

    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["quarter"] = df["date"].dt.quarter
    # Maha: Oct-Apr, Yala: May-Sep (Weerasekara et al., 2026)
    df["is_maha_season"] = df["month"].apply(lambda m: 1 if (m >= 10 or m <= 4) else 0)

    for col in EXOG_COLUMNS:
        df[f"{col}_lag_1"] = df[col].shift(1)

    df["target"] = df[target_col]
    df = df.dropna().reset_index(drop=True)
    return df


def get_feature_columns(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in NON_FEATURE_COLUMNS]
