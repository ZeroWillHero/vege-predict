"""Train an LSTM model per vegetable on the featurized price + exogenous sequence."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.models.common import parse_vegetable_arg, run_for_vegetables
from src.utils.io import get_model_dir, load_config

MODEL_FAMILY = "lstm"


def _scale(df: pd.DataFrame, feature_cols: list, scaler: StandardScaler | None = None):
    cols = ["target"] + feature_cols
    values = df[cols].values.astype("float32")
    if scaler is None:
        scaler = StandardScaler().fit(values)
    return scaler.transform(values), scaler


def _build_sequences(scaled_values: np.ndarray, lookback: int):
    X, y = [], []
    for i in range(lookback, len(scaled_values)):
        X.append(scaled_values[i - lookback : i])
        y.append(scaled_values[i, 0])  # target is column 0
    return np.array(X), np.array(y)


def _inverse_target(y_scaled: np.ndarray, scaler: StandardScaler) -> np.ndarray:
    return y_scaled * scaler.scale_[0] + scaler.mean_[0]


def _build_model(n_features: int, config: dict):
    from tensorflow import keras

    params = config["models"]["lstm"]
    model = keras.Sequential()
    for i in range(params["layers"]):
        return_sequences = i < params["layers"] - 1
        kwargs = {"input_shape": (params["lookback"], n_features)} if i == 0 else {}
        model.add(keras.layers.LSTM(params["units"], return_sequences=return_sequences, **kwargs))
        model.add(keras.layers.Dropout(params["dropout"]))
    model.add(keras.layers.Dense(1))
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=params["learning_rate"]), loss="mse")
    return model


def _fit_lstm(train_df: pd.DataFrame, feature_cols: list, config: dict):
    from tensorflow import keras

    lookback = config["models"]["lstm"]["lookback"]
    scaled_train, scaler = _scale(train_df, feature_cols)
    X_train, y_train = _build_sequences(scaled_train, lookback)

    model = _build_model(len(feature_cols) + 1, config)
    early_stop = keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)
    model.fit(
        X_train,
        y_train,
        epochs=config["models"]["lstm"]["epochs"],
        batch_size=config["models"]["lstm"]["batch_size"],
        callbacks=[early_stop],
        verbose=0,
    )
    return model, scaler


def _predict(model, scaler, train_df, test_df, feature_cols, lookback):
    combined = pd.concat([train_df.tail(lookback), test_df])
    scaled_combined, _ = _scale(combined, feature_cols, scaler=scaler)
    X_test, _ = _build_sequences(scaled_combined, lookback)
    preds_scaled = model.predict(X_test, verbose=0).flatten()
    return _inverse_target(preds_scaled, scaler)


def make_fit_predict(config):
    lookback = config["models"]["lstm"]["lookback"]

    def fit_predict(train_df, test_df, feature_cols):
        model, scaler = _fit_lstm(train_df, feature_cols, config)
        return _predict(model, scaler, train_df, test_df, feature_cols, lookback)

    return fit_predict


def make_fit_final(config):
    def fit_final(full_df, feature_cols, vegetable):
        model, scaler = _fit_lstm(full_df, feature_cols, config)
        # save_model() only persists the keras model itself; the scaler (needed to
        # un-scale predictions at inference time) is saved alongside it here.
        model_dir = get_model_dir(MODEL_FAMILY, config)
        model_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump({"scaler": scaler, "feature_cols": feature_cols}, model_dir / f"{vegetable}_scaler.pkl")
        return model

    return fit_final


def train_all_vegetables(config: dict, vegetables=None) -> list:
    vegetables = vegetables or config["vegetables"]
    return run_for_vegetables(
        vegetables, MODEL_FAMILY, config, make_fit_predict(config), make_fit_final(config), keras=True
    )


def main():
    args = parse_vegetable_arg()
    config = load_config()
    vegetables = [args.vegetable] if args.vegetable else None
    train_all_vegetables(config, vegetables)


if __name__ == "__main__":
    main()
