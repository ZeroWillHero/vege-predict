"""Fit SARIMAX for the linear/seasonal component, then XGBoost on SARIMAX residuals."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
warnings.filterwarnings("ignore")

from xgboost import XGBRegressor

from src.models.common import parse_vegetable_arg, run_for_vegetables
from src.models.sarimax.train import SarimaxArtifact, get_order_for_vegetable
from src.models.sarimax.train import _fit as fit_sarimax
from src.utils.io import load_config

MODEL_FAMILY = "hybrid_xgboost_sarimax"


class HybridSarimaxResidualModel:
    """Bundles a lightweight SARIMAX artifact with an ML model trained on its residuals.

    Uses SarimaxArtifact (params + small history frame) rather than the raw fitted
    SARIMAX results object, which otherwise balloons the pickled artifact to 100s of MB
    (see src/models/sarimax/train.py for why).
    """

    def __init__(self, sarimax_artifact: SarimaxArtifact, residual_model, feature_cols):
        self.sarimax_artifact = sarimax_artifact
        self.residual_model = residual_model
        self.feature_cols = feature_cols

    def predict(self, future_df):
        sarimax_forecast = self.sarimax_artifact.forecast(future_df)
        residual_pred = self.residual_model.predict(future_df[self.feature_cols])
        return sarimax_forecast + residual_pred


def _make_residual_model(config):
    params = config["models"]["xgboost"]
    return XGBRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        learning_rate=params["learning_rate"],
        random_state=42,
        n_jobs=-1,
    )


def _fit_hybrid(train_df, feature_cols, config, order):
    exog_cols = config["models"]["sarimax"]["exog_features"]
    seasonal_order = tuple(config["models"]["sarimax"]["seasonal_order"])

    sarimax_result = fit_sarimax(train_df, config, order=order)
    train_fitted = sarimax_result.get_prediction().predicted_mean.values
    residuals = train_df["target"].values - train_fitted
    sarimax_artifact = SarimaxArtifact(sarimax_result.params, order, seasonal_order, exog_cols, train_df)

    residual_model = _make_residual_model(config)
    residual_model.fit(train_df[feature_cols], residuals)

    return HybridSarimaxResidualModel(sarimax_artifact, residual_model, feature_cols)


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
