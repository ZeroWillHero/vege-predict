"""Train a Random Forest regressor per vegetable using lag/rolling/calendar features."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent.parent))

from sklearn.ensemble import RandomForestRegressor

from src.models.common import parse_vegetable_arg, run_for_vegetables
from src.utils.io import load_config

MODEL_FAMILY = "random_forest"


def _make_model(config):
    params = config["models"]["random_forest"]
    return RandomForestRegressor(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        random_state=42,
        n_jobs=-1,
    )


def make_fit_predict(config):
    def fit_predict(train_df, test_df, feature_cols):
        model = _make_model(config)
        model.fit(train_df[feature_cols], train_df["target"])
        return model.predict(test_df[feature_cols])

    return fit_predict


def make_fit_final(config):
    def fit_final(full_df, feature_cols, vegetable):
        model = _make_model(config)
        model.fit(full_df[feature_cols], full_df["target"])
        return model

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
