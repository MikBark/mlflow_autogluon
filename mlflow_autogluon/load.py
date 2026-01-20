"""Model loading functionality for AutoGluon MLflow flavor."""

from __future__ import annotations

from typing import Any

from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.models import Model

from mlflow_autogluon.constants import (
    AUTODEPLOY_SUBPATH,
    FLAVOR_NAME,
)


def load_model(
    model_uri: str,
    dst_path: str | None = None,
) -> Any | object:
    """
    Load an AutoGluon model from MLflow.

    Args:
        model_uri: URI pointing to the model (e.g., 'runs:/<run_id>/model')
        dst_path: Optional local path to download model to

    Returns:
        Loaded AutoGluon model instance

    Raises:
        MlflowException: If model cannot be loaded or flavor configuration is invalid
    """
    import os

    local_model_path = download_artifacts(artifact_uri=model_uri, dst_path=dst_path)

    model = Model.load(local_model_path)

    flavor_conf = model.flavors[FLAVOR_NAME]

    model_type = flavor_conf.get("model_type", "tabular")

    autogluon_model_path = os.path.join(local_model_path, AUTODEPLOY_SUBPATH)

    if model_type == "tabular":
        from autogluon.tabular import TabularPredictor

        return TabularPredictor.load(autogluon_model_path)
    elif model_type == "multimodal":
        from autogluon.multimodal import MultiModalPredictor

        return MultiModalPredictor.load(autogluon_model_path)
    elif model_type == "vision":
        from autogluon.vision import VisionPredictor

        return VisionPredictor.load(autogluon_model_path)
    elif model_type == "timeseries":
        from autogluon.timeseries import TimeSeriesPredictor

        return TimeSeriesPredictor.load(autogluon_model_path)
    else:
        raise MlflowException(
            message=f"Unsupported model_type '{model_type}' in flavor configuration"
        )


def _load_pyfunc(path: str) -> Any:
    """
    Load AutoGluon model as PyFunc.

    This is used internally by MLflow when loading model with pyfunc flavor.

    Args:
        path: Local path to model directory

    Returns:
        PyFunc-compatible wrapper instance
    """
    from mlflow_autogluon.pyfunc import _AutoGluonModelWrapper

    return _AutoGluonModelWrapper(path)
