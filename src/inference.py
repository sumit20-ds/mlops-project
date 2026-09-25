"""Shared inference logic — one code path for FastAPI and SageMaker."""
from __future__ import annotations

import functools
import logging

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from src import config

logger = logging.getLogger(__name__)


class ModelNotLoadedError(RuntimeError):
    pass


@functools.lru_cache(maxsize=1)
def load_model(model_path: str | None = None) -> Pipeline:
    """Load and cache the trained sklearn pipeline.

    NOTE: this model was trained with scikit-learn==1.6.1. Loading it with
    a mismatched sklearn version can raise AttributeError/UnpicklingError
    on newer releases (confirmed with 1.8.0) because internal transformer
    classes changed between versions. requirements.txt and the Dockerfile
    pin scikit-learn==1.6.1 for exactly this reason — do not bump it
    without re-exporting the model from a matching training environment.
    """
    path = model_path or str(config.MODEL_PATH)
    try:
        model = joblib.load(path)
    except FileNotFoundError as exc:
        raise ModelNotLoadedError(f"Model file not found at {path}") from exc
    except Exception as exc:  # noqa: BLE001 - surface version-mismatch errors clearly
        raise ModelNotLoadedError(
            f"Failed to load model at {path}. This is almost always a "
            f"scikit-learn version mismatch — this model requires "
            f"scikit-learn==1.6.1. Original error: {exc}"
        ) from exc
    logger.info("Loaded model from %s", path)
    return model


def rows_to_dataframe(rows: list[dict]) -> pd.DataFrame:
    """Convert a list of feature dicts into the exact column order/dtype
    the pipeline's ColumnTransformer was fit on."""
    df = pd.DataFrame(rows)
    missing = set(config.FEATURE_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required features: {sorted(missing)}")
    return df[config.FEATURE_COLUMNS]


def predict(rows: list[dict], model_path: str | None = None) -> list[float]:
    model = load_model(model_path)
    df = rows_to_dataframe(rows)
    preds = model.predict(df)
    return [round(float(p), 3) for p in preds]
