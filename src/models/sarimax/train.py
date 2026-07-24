"""Train a SARIMAX model per vegetable using weather/fuel as exogenous regressors."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
warnings.filterwarnings("ignore")

from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.models.common import parse_vegetable_arg, run_for_vegetables
from src.utils.io import load_config

MODEL_FAMILY = "sarimax"


class SarimaxArtifact:
    """Lightweight, picklable SARIMAX artifact.

    Pickling a fitted SARIMAXResultsWrapper directly balloons to 100s of MB, because
    it carries the full Kalman filter state/covariance history (large here due to the
    seasonal_order period of 52). This stores just the fitted params plus the small
    training frame needed to reconstruct the filter (verified to reproduce identical
    forecasts, at ~20KB instead of ~320MB).
    """

    def __init__(self, params, order, seasonal_order, exog_cols, history_df):
        self.params = params
        self.order = order
        self.seasonal_order = seasonal_order
        self.exog_cols = exog_cols
        self.history_df = history_df[["target"] + exog_cols].copy()

    def _reconstruct(self):
        model = SARIMAX(
            self.history_df["target"],
            exog=self.history_df[self.exog_cols],
            order=self.order,
            seasonal_order=self.seasonal_order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        return model.filter(self.params)

    def forecast(self, future_df):
        result = self._reconstruct()
        fc = result.get_forecast(steps=len(future_df), exog=future_df[self.exog_cols])
        return fc.predicted_mean.values


def _fit(train_df, config):
    exog_cols = config["models"]["sarimax"]["exog_features"]
    order = tuple(config["models"]["sarimax"]["order"])
    seasonal_order = tuple(config["models"]["sarimax"]["seasonal_order"])
    # NOTE: simple_differencing=True must NOT be combined with exog here — it breaks
    # out-of-sample exog extension in statsmodels and produces divergent forecasts
    # (confirmed empirically: predictions swung to +/-700 on a 100-900 price series).
    model = SARIMAX(
        train_df["target"],
        exog=train_df[exog_cols],
        order=order,
        seasonal_order=seasonal_order,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    return model.fit(disp=False, maxiter=200)


def make_fit_predict(config):
    exog_cols = config["models"]["sarimax"]["exog_features"]

    def fit_predict(train_df, test_df, feature_cols):
        res = _fit(train_df, config)
        forecast = res.get_forecast(steps=len(test_df), exog=test_df[exog_cols])
        return forecast.predicted_mean.values

    return fit_predict


def make_fit_final(config):
    exog_cols = config["models"]["sarimax"]["exog_features"]
    order = tuple(config["models"]["sarimax"]["order"])
    seasonal_order = tuple(config["models"]["sarimax"]["seasonal_order"])

    def fit_final(full_df, feature_cols, vegetable):
        res = _fit(full_df, config)
        return SarimaxArtifact(res.params, order, seasonal_order, exog_cols, full_df)

    return fit_final


def train_all_vegetables(config: dict, vegetables=None) -> list:
    vegetables = vegetables or config["vegetables"]
    return run_for_vegetables(vegetables, MODEL_FAMILY, config, make_fit_predict(config), make_fit_final(config))


def main():
    args = parse_vegetable_arg()
    config = load_config()
    vegetables = [args.vegetable] if args.vegetable else None
    train_all_vegetables(config, vegetables)


if __name__ == "__main__":
    main()
