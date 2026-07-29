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
    elif hasattr(model, "save_artifact"):
        # Composite artifact (part-Keras, part-pickle) that knows how to split itself
        # across multiple files given an extensionless base path — e.g. HybridSarimaxLstmModel
        # in src/models/hybrid_lstm_sarimax/train.py, since a Keras model can't share one
        # joblib-picklable wrapper with a SarimaxArtifact/sklearn object.
        path = model_dir / vegetable
        model.save_artifact(path)
    else:
        path = model_dir / f"{vegetable}.pkl"
        joblib.dump(model, path)
    return path


def load_model(model_family: str, vegetable: str, config: dict, keras: bool = False, artifact_cls=None):
    model_dir = get_model_dir(model_family, config)
    if keras:
        from tensorflow import keras as tf_keras

        return tf_keras.models.load_model(model_dir / f"{vegetable}.keras")
    if artifact_cls is not None:
        return artifact_cls.load_artifact(model_dir / vegetable)
    return joblib.load(model_dir / f"{vegetable}.pkl")
