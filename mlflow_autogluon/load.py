"""Model loading functionality for AutoGluon MLflow flavor."""

from __future__ import annotations

import os
from typing import Any

from mlflow.artifacts import download_artifacts
from mlflow.models import Model

from mlflow_autogluon.constants import (
    AUTODEPLOY_SUBPATH,
    FLAVOR_NAME,
)
from mlflow_autogluon.domain.model_type import ModelType
from mlflow_autogluon.domain.predictor_loader import get_loader


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
    local_model_path = download_artifacts(artifact_uri=model_uri, dst_path=dst_path)

    model = Model.load(local_model_path)

    flavor_conf = model.flavors[FLAVOR_NAME]

    model_type = ModelType.from_string(
        flavor_conf.get("model_type", ModelType.TABULAR.value)
    )

    autogluon_model_path = os.path.join(local_model_path, AUTODEPLOY_SUBPATH)

    loader = get_loader(model_type)
    return loader.load(autogluon_model_path)


def _load_pyfunc(path: str) -> Any:
    """
    Load AutoGluon model as PyFunc.

    This is used internally by MLflow when loading model with pyfunc flavor.

    Args:
        path: Local path to model directory

    Returns:
        PyFunc-compatible wrapper instance
    """
    from mlflow_autogluon.pyfunc.pyfunc import _AutoGluonModelWrapper

    return _AutoGluonModelWrapper(path)
