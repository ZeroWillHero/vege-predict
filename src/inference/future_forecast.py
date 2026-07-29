"""Genuine future price forecasting: chains each model family's own one-step-ahead design
recursively, N weeks past the last available price date.

Reuses every model family's already-fitted, already-persisted production artifact
(trained_models/<family>/<vegetable>.*, written by scripts/train_all.py) — no retraining.

Gotcha (see CLAUDE.md): SARIMAX-based families must be forecast with ONE call covering the
whole horizon (SarimaxArtifact.forecast() / hybrid.sarimax_artifact.forecast()), never once
per week with only that week's own exog row — the latter would incorrectly treat every future
week as if it were the single next week after training ends, rather than N steps ahead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import joblib
import numpy as np
import pandas as pd

from scripts.fetch_weather_openmeteo import DISTRICTS, climatology_estimate, fetch_district_forecast, to_weekly
from src.evaluation.metrics import prediction_interval
from src.features.feature_engineering import EXOG_COLUMNS, LAG_WEEKS, ROLLING_WINDOWS
from src.models.common import load_features
from src.models.hybrid_lstm_sarimax.train import HybridSarimaxLstmModel
from src.models.lstm.train import _predict as _lstm_predict
from src.utils.io import PROJECT_ROOT, get_model_dir, get_processed_path, load_model

SARIMAX_HYBRID_FAMILIES = {
    "hybrid_xgboost_sarimax",
    "hybrid_catboost_sarimax",
    "hybrid_random_forest_sarimax",
    "hybrid_lstm_sarimax",
}
TABULAR_FAMILIES = {"random_forest", "xgboost", "catboost"}


def build_future_exog(vegetable: str, config: dict, horizon_weeks: int | None = None) -> pd.DataFrame:
    """[date, average_temperature, average_rainfall, diesel_price, weather_source] reaching
    `horizon_weeks` weeks past *today's* real date (not just past the last known price date —
    the price data can lag behind real time by weeks, so anchoring purely to it would produce
    "future" weeks that are already in the past by the time this actually runs). Any gap between
    the last known price week and today is bridged automatically (recursive lag features can't
    skip it), so the returned range is [last_known_date + 1 week, today + horizon_weeks],
    week-aligned to the same Monday-start convention as the rest of the pipeline.

    Weather/diesel are resolved in three tiers, cheapest/most-accurate first: (1) already-
    collected raw weather.csv/fuel_data_weekly.csv, which are refreshed independently of price
    data and often already run ahead of it; (2) Open-Meteo's real forecast API (~16 days out),
    fetched lazily only if tier 1 doesn't cover a week; (3) climatology (that ISO week's
    historical average) for anything further out."""
    horizon_weeks = horizon_weeks or config["forecasting"]["future_horizon_weeks"]
    processed = pd.read_csv(get_processed_path(vegetable, config), parse_dates=["date"])
    last_date = processed["date"].max()

    today = pd.Timestamp.today().normalize()
    today_week_start = today - pd.Timedelta(days=today.weekday())  # Monday-align, matches week_start convention
    weeks_to_bridge = max(0, (today_week_start - last_date).days // 7)
    total_weeks = weeks_to_bridge + horizon_weeks
    future_dates = [last_date + pd.Timedelta(weeks=i) for i in range(1, total_weeks + 1)]

    district = config["weather_district_map"][vegetable]
    lat, lon = DISTRICTS[district]

    weather_raw = pd.read_csv(
        PROJECT_ROOT / config["data"]["raw_weather_dir"] / "weather.csv", parse_dates=["week_start_date"]
    )
    weather_known = weather_raw[weather_raw["district"] == district].set_index("week_start_date")

    fuel_raw = pd.read_csv(PROJECT_ROOT / config["data"]["raw_fuel_dir"] / "fuel_data_weekly.csv", parse_dates=["date"])
    fuel_known = fuel_raw.set_index("date")["diesel_price"].sort_index()
    last_diesel = float(fuel_known.iloc[-1])

    forecast_weekly = None  # fetched lazily, at most once, only if tier 1 doesn't cover everything

    rows = []
    for d in future_dates:
        d_ts = pd.Timestamp(d)
        if d_ts in weather_known.index:
            temp = float(weather_known.loc[d_ts, "average_temperature"])
            rain = float(weather_known.loc[d_ts, "average_rainfall"])
            source = "known"
        else:
            if forecast_weekly is None:
                daily = fetch_district_forecast(district, lat, lon, days=16)
                forecast_weekly = to_weekly(daily).set_index("week_start_date")
            d_key = d_ts.date()
            if d_key in forecast_weekly.index:
                temp = float(forecast_weekly.loc[d_key, "average_temperature"])
                rain = float(forecast_weekly.loc[d_key, "average_rainfall"])
                source = "forecast"
            else:
                iso_week = int(d_ts.isocalendar().week)
                temp, rain = climatology_estimate(district, iso_week)
                source = "climatology"
        diesel = float(fuel_known.get(d_ts, last_diesel))
        rows.append(
            {
                "date": d,
                "average_temperature": round(temp, 2),
                "average_rainfall": round(rain, 2),
                "diesel_price": diesel,
                "weather_source": source,
            }
        )
    return pd.DataFrame(rows)


def _build_future_row(history: pd.DataFrame, future_date: pd.Timestamp, exog_now: pd.Series) -> pd.DataFrame:
    """One feature row for `future_date`, matching feature_engineering.py's formulas exactly.
    `history` has columns [date, price] + EXOG_COLUMNS, containing every week strictly before
    `future_date` (actual history + already-predicted future weeks appended so far)."""
    price = history["price"]
    row = {}
    for lag in LAG_WEEKS:
        row[f"price_lag_{lag}"] = price.iloc[-lag]
    for window in ROLLING_WINDOWS:
        tail = price.iloc[-window:]
        row[f"price_rolling_mean_{window}"] = tail.mean()
        row[f"price_rolling_std_{window}"] = tail.std()
    row["month"] = future_date.month
    row["week_of_year"] = int(future_date.isocalendar().week)
    row["quarter"] = future_date.quarter
    row["is_maha_season"] = 1 if (future_date.month >= 10 or future_date.month <= 4) else 0
    for col in EXOG_COLUMNS:
        row[col] = exog_now[col]
        row[f"{col}_lag_1"] = history[col].iloc[-1]
    row["target"] = 0.0  # placeholder — never read; see module docstring / LSTM callers below
    row["date"] = future_date
    return pd.DataFrame([row])


def _append_history(history: pd.DataFrame, future_date: pd.Timestamp, price: float, exog_now: pd.Series) -> pd.DataFrame:
    new_row = {"date": future_date, "price": price}
    for col in EXOG_COLUMNS:
        new_row[col] = exog_now[col]
    return pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)


def _holdout_interval(vegetable: str, model_family: str, config: dict) -> tuple[float, float]:
    path = PROJECT_ROOT / "results" / "metrics" / "holdout_predictions.csv"
    df = pd.read_csv(path)
    mask = (df["vegetable"] == vegetable) & (df["model"] == model_family)
    residuals = (df.loc[mask, "actual"] - df.loc[mask, "predicted"]).values
    confidence = config["forecasting"].get("interval_confidence", 0.8)
    return prediction_interval(residuals, confidence)


def _package(vegetable: str, model_family: str, dates: list, preds: np.ndarray, weather_source: pd.Series, lower_off: float, upper_off: float) -> pd.DataFrame:
    preds = np.asarray(preds, dtype=float)
    return pd.DataFrame(
        {
            "vegetable": vegetable,
            "model": model_family,
            "date": dates,
            "predicted": preds,
            "predicted_lower": preds + lower_off,
            "predicted_upper": preds + upper_off,
            "weather_source": weather_source.values,
        }
    )


def recursive_forecast(vegetable: str, model_family: str, config: dict, horizon_weeks: int | None = None) -> pd.DataFrame:
    """Genuine future forecast for one vegetable/model family, `horizon_weeks` weeks past the
    last known price date. Returns a frame shaped like holdout_predictions.csv (no `actual`
    column — genuinely unknown)."""
    exog_df = build_future_exog(vegetable, config, horizon_weeks)
    future_dates = [pd.Timestamp(d) for d in exog_df["date"]]
    exog_cols_sarimax = config["models"]["sarimax"]["exog_features"]
    lower_off, upper_off = _holdout_interval(vegetable, model_family, config)

    processed = pd.read_csv(get_processed_path(vegetable, config), parse_dates=["date"])
    target_col = config["forecasting"]["target_column"]
    history = processed[["date", target_col] + EXOG_COLUMNS].rename(columns={target_col: "price"}).copy()

    if model_family == "sarimax":
        model = load_model("sarimax", vegetable, config)
        preds = model.forecast(exog_df[exog_cols_sarimax])
        return _package(vegetable, model_family, future_dates, preds, exog_df["weather_source"], lower_off, upper_off)

    if model_family in SARIMAX_HYBRID_FAMILIES:
        if model_family == "hybrid_lstm_sarimax":
            hybrid = load_model(model_family, vegetable, config, artifact_cls=HybridSarimaxLstmModel)
        else:
            hybrid = load_model(model_family, vegetable, config)
        sarimax_component = hybrid.sarimax_artifact.forecast(exog_df[exog_cols_sarimax])

        preds = []
        resid_hist = hybrid.resid_history_tail_df.copy() if model_family == "hybrid_lstm_sarimax" else None
        for i, d in enumerate(future_dates):
            row = _build_future_row(history, d, exog_df.iloc[i])
            if model_family == "hybrid_lstm_sarimax":
                resid_row = row[hybrid.feature_cols].copy()
                resid_row["target"] = 0.0
                resid_pred = _lstm_predict(hybrid.lstm_model, hybrid.scaler, resid_hist, resid_row, hybrid.feature_cols, hybrid.lookback)[0]
                new_resid_row = row[hybrid.feature_cols].copy()
                new_resid_row["target"] = resid_pred
                resid_hist = pd.concat([resid_hist, new_resid_row], ignore_index=True).tail(hybrid.lookback)
            else:
                resid_pred = hybrid.residual_model.predict(row[hybrid.feature_cols])[0]
            price = float(sarimax_component[i] + resid_pred)
            preds.append(price)
            history = _append_history(history, d, price, exog_df.iloc[i])
        return _package(vegetable, model_family, future_dates, preds, exog_df["weather_source"], lower_off, upper_off)

    if model_family == "lstm":
        model = load_model("lstm", vegetable, config, keras=True)
        sidecar = joblib.load(get_model_dir("lstm", config) / f"{vegetable}_scaler.pkl")
        scaler, feature_cols = sidecar["scaler"], sidecar["feature_cols"]
        lookback = config["models"]["lstm"]["lookback"]
        feat_all, _ = load_features(vegetable, config)
        lstm_hist = feat_all[["target"] + feature_cols].tail(lookback).copy()

        preds = []
        for i, d in enumerate(future_dates):
            row = _build_future_row(history, d, exog_df.iloc[i])
            lstm_row = row[feature_cols].copy()
            lstm_row["target"] = 0.0
            price = float(_lstm_predict(model, scaler, lstm_hist, lstm_row, feature_cols, lookback)[0])
            preds.append(price)
            new_row = row[feature_cols].copy()
            new_row["target"] = price
            lstm_hist = pd.concat([lstm_hist, new_row], ignore_index=True).tail(lookback)
            history = _append_history(history, d, price, exog_df.iloc[i])
        return _package(vegetable, model_family, future_dates, preds, exog_df["weather_source"], lower_off, upper_off)

    if model_family in TABULAR_FAMILIES:
        model = load_model(model_family, vegetable, config)
        feat_all, feature_cols = load_features(vegetable, config)
        preds = []
        for i, d in enumerate(future_dates):
            row = _build_future_row(history, d, exog_df.iloc[i])
            price = float(model.predict(row[feature_cols])[0])
            preds.append(price)
            history = _append_history(history, d, price, exog_df.iloc[i])
        return _package(vegetable, model_family, future_dates, preds, exog_df["weather_source"], lower_off, upper_off)

    raise ValueError(f"Unknown model family: {model_family}")
