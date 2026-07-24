"""Fit SARIMAX for the linear/seasonal component, then CatBoost on SARIMAX residuals."""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))
warnings.filterwarnings("ignore")

from catboost import CatBoostRegressor

from src.models.common import parse_vegetable_arg, run_for_vegetables
from src.models.hybrid_xgboost_sarimax.train import HybridSarimaxResidualModel
from src.models.sarimax.train import SarimaxArtifact
from src.models.sarimax.train import _fit as fit_sarimax
from src.utils.io import load_config

MODEL_FAMILY = "hybrid_catboost_sarimax"


def _make_residual_model(config):
    params = config["models"]["catboost"]
    return CatBoostRegressor(
        iterations=params["iterations"],
        depth=params["depth"],
        learning_rate=params["learning_rate"],
        random_seed=42,
        verbose=False,
    )


def _fit_hybrid(train_df, feature_cols, config):
    exog_cols = config["models"]["sarimax"]["exog_features"]
    order = tuple(config["models"]["sarimax"]["order"])
    seasonal_order = tuple(config["models"]["sarimax"]["seasonal_order"])

    sarimax_result = fit_sarimax(train_df, config)
    train_fitted = sarimax_result.get_prediction().predicted_mean.values
    residuals = train_df["target"].values - train_fitted
    sarimax_artifact = SarimaxArtifact(sarimax_result.params, order, seasonal_order, exog_cols, train_df)

    residual_model = _make_residual_model(config)
    residual_model.fit(train_df[feature_cols], residuals)

    return HybridSarimaxResidualModel(sarimax_artifact, residual_model, feature_cols)


def make_fit_predict(config):
    def fit_predict(train_df, test_df, feature_cols):
        hybrid = _fit_hybrid(train_df, feature_cols, config)
        return hybrid.predict(test_df)

    return fit_predict


def make_fit_final(config):
    def fit_final(full_df, feature_cols, vegetable):
        return _fit_hybrid(full_df, feature_cols, config)

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
