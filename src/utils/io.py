"""Config loading and model save/load helpers shared across training scripts."""

from pathlib import Path

import joblib
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def get_processed_path(vegetable: str, config: dict) -> Path:
    return PROJECT_ROOT / config["data"]["processed_dir"] / f"{vegetable}.csv"


def get_model_dir(model_family: str, config: dict) -> Path:
    return PROJECT_ROOT / config["paths"]["trained_models_dir"] / model_family


def save_model(model, model_family: str, vegetable: str, config: dict, keras: bool = False) -> Path:
    model_dir = get_model_dir(model_family, config)
    model_dir.mkdir(parents=True, exist_ok=True)
    if keras:
        path = model_dir / f"{vegetable}.keras"
        model.save(path)
    else:
        path = model_dir / f"{vegetable}.pkl"
        joblib.dump(model, path)
    return path


def load_model(model_family: str, vegetable: str, config: dict, keras: bool = False):
    model_dir = get_model_dir(model_family, config)
    if keras:
        from tensorflow import keras as tf_keras

        return tf_keras.models.load_model(model_dir / f"{vegetable}.keras")
    return joblib.load(model_dir / f"{vegetable}.pkl")
