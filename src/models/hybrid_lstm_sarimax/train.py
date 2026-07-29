"""Fit SARIMAX for the linear/seasonal component, then LSTM on SARIMAX residuals."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
warnings.filterwarnings("ignore")

import joblib
import numpy as np

from src.models.common import parse_vegetable_arg, run_for_vegetables
from src.models.lstm.train import _fit_lstm, _predict
from src.models.sarimax.train import SarimaxArtifact, get_order_for_vegetable
from src.models.sarimax.train import _fit as fit_sarimax
from src.utils.io import load_config

MODEL_FAMILY = "hybrid_lstm_sarimax"


class HybridSarimaxLstmModel:
    """Bundles a lightweight SARIMAX artifact with an LSTM trained on its residuals.

    A SarimaxArtifact and a Keras model can't share one joblib-picklable wrapper object
    (the same reason src/models/lstm/train.py saves its scaler as a separate sidecar file).
    save_artifact/load_artifact split this across a `.keras` file and a `.pkl` file — see
    save_model()/load_model() in src/utils/io.py for the dispatch that calls these.
    """

    def __init__(self, sarimax_artifact, lstm_model, scaler, feature_cols, lookback, resid_history_tail_df, resid_clip):
        self.sarimax_artifact = sarimax_artifact
        self.lstm_model = lstm_model
        self.scaler = scaler
        self.feature_cols = feature_cols
        self.lookback = lookback
        # Trailing `lookback` rows of ["target"] + feature_cols, where "target" here is the
        # in-sample SARIMAX residual (not price) — gives predict() the context to build
        # sequences for the first rows of a future test set.
        self.resid_history_tail_df = resid_history_tail_df
        # SARIMAX's out-of-sample forecast can occasionally diverge to extreme values on a
        # small/early training window (a known SARIMAX fragility in this project — see the
        # simple_differencing gotcha in CLAUDE.md; confirmed empirically here too, e.g.
        # cabbage's second CV fold). The tree-based residual hybrids tolerate that silently
        # (one bad-but-finite fold score); the LSTM path doesn't, because the residual gets
        # cast to float32 for scaling and an extreme-enough value overflows to inf, which
        # StandardScaler rejects outright. Clipping to a generous multiple of the in-sample
        # residual scale keeps the pipeline numerically stable without pretending the
        # underlying SARIMAX forecast for that fold was good.
        self.resid_clip = resid_clip

    def predict(self, future_df):
        sarimax_forecast = self.sarimax_artifact.forecast(future_df)
        resid_future_df = future_df[self.feature_cols].copy()
        raw_residual = future_df["target"].values - sarimax_forecast
        resid_future_df["target"] = np.clip(raw_residual, -self.resid_clip, self.resid_clip)
        residual_pred = _predict(
            self.lstm_model, self.scaler, self.resid_history_tail_df, resid_future_df, self.feature_cols, self.lookback
        )
        return sarimax_forecast + residual_pred

    def save_artifact(self, base_path: Path):
        base_path.parent.mkdir(parents=True, exist_ok=True)
        self.lstm_model.save(base_path.with_suffix(".keras"))
        joblib.dump(
            {
                "sarimax_artifact": self.sarimax_artifact,
                "scaler": self.scaler,
                "feature_cols": self.feature_cols,
                "lookback": self.lookback,
                "resid_history_tail_df": self.resid_history_tail_df,
                "resid_clip": self.resid_clip,
            },
            base_path.with_suffix(".pkl"),
        )

    @classmethod
    def load_artifact(cls, base_path: Path):
        from tensorflow import keras as tf_keras

        lstm_model = tf_keras.models.load_model(base_path.with_suffix(".keras"))
        state = joblib.load(base_path.with_suffix(".pkl"))
        return cls(
            state["sarimax_artifact"],
            lstm_model,
            state["scaler"],
            state["feature_cols"],
            state["lookback"],
            state["resid_history_tail_df"],
            state["resid_clip"],
        )


def _fit_hybrid(train_df, feature_cols, config, order):
    exog_cols = config["models"]["sarimax"]["exog_features"]
    seasonal_order = tuple(config["models"]["sarimax"]["seasonal_order"])
    lookback = config["models"]["lstm"]["lookback"]

    sarimax_result = fit_sarimax(train_df, config, order=order)
    train_fitted = sarimax_result.get_prediction().predicted_mean.values
    residuals = train_df["target"].values - train_fitted
    sarimax_artifact = SarimaxArtifact(sarimax_result.params, order, seasonal_order, exog_cols, train_df)

    # See HybridSarimaxLstmModel.resid_clip: bounds how far an out-of-sample SARIMAX
    # forecast is allowed to pull the "residual" before it reaches the LSTM's scaler.
    resid_clip = float(np.std(residuals) * 10)

    # Reuse the plain LSTM's fit routine unchanged by handing it a frame where "target"
    # is the SARIMAX residual instead of price — _fit_lstm has no hardcoded notion of what
    # "target" represents.
    resid_train_df = train_df[feature_cols].copy()
    resid_train_df["target"] = residuals
    lstm_model, scaler = _fit_lstm(resid_train_df, feature_cols, config)
    resid_history_tail_df = resid_train_df[["target"] + feature_cols].tail(lookback).copy()

    return HybridSarimaxLstmModel(
        sarimax_artifact, lstm_model, scaler, feature_cols, lookback, resid_history_tail_df, resid_clip
    )


def make_fit_predict(config, order):
    def fit_predict(train_df, test_df, feature_cols):
        hybrid = _fit_hybrid(train_df, feature_cols, config, order)
        return hybrid.predict(test_df)

    return fit_predict


def make_fit_final(config, order):
    def fit_final(full_df, feature_cols, vegetable):
        return _fit_hybrid(full_df, feature_cols, config, order)

    return fit_final


def train_all_vegetables(config: dict, vegetables=None):
    vegetables = vegetables or config["vegetables"]
    results, predictions = [], []
    for vegetable in vegetables:
        order = get_order_for_vegetable(vegetable, config)
        r, p = run_for_vegetables(
            [vegetable], MODEL_FAMILY, config, make_fit_predict(config, order), make_fit_final(config, order)
        )
        results.extend(r)
        predictions.extend(p)
    return results, predictions


def main():
    args = parse_vegetable_arg()
    config = load_config()
    vegetables = [args.vegetable] if args.vegetable else None
    train_all_vegetables(config, vegetables)


if __name__ == "__main__":
    main()
