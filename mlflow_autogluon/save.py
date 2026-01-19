"""Model saving functionality for AutoGluon MLflow flavor."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE

from mlflow_autogluon.constants import (
    AUTODEPLOY_METADATA_FILE,
    AUTODEPLOY_SUBPATH,
    FLAVOR_NAME,
)
from mlflow_autogluon.requirements import get_default_conda_env


def save_model(
    autogluon_model: Any | object,
    path: str,
    model_type: str = "tabular",
    mlflow_model: Model | None = None,
    conda_env: dict | str | None = None,
    pip_requirements: list[str] | None = None,
    extra_pip_requirements: list[str] | None = None,
    **kwargs: Any,
) -> None:
    """
    Save an AutoGluon model to a path on the local file system.

    Args:
        autogluon_model: AutoGluon model instance (e.g., TabularPredictor)
        path: Local path where model is to be saved
        model_type: Type of AutoGluon model ('tabular', 'multimodal', etc.)
        mlflow_model: MLflow model config to add to (creates new if None)
        conda_env: Conda environment dict or path to conda env yaml file
        pip_requirements: Override default pip requirements
        extra_pip_requirements: Extra pip requirements to add to defaults
        **kwargs: Additional arguments for AutoGluon-specific configuration

    Raises:
        MlflowException: If model_type is not supported or model lacks save() method
    """
    import json
    import shutil

    supported_types = ["tabular", "multimodal", "vision", "timeseries"]
    if model_type not in supported_types:
        msg = f"Unsupported model_type '{model_type}'. Supported types: {supported_types}"
        raise MlflowException(
            invalid_parameter_value=INVALID_PARAMETER_VALUE,
            message=msg,
        )

    if not hasattr(autogluon_model, "save"):
        raise MlflowException(
            message=(
                f"Model of type '{type(autogluon_model).__name__}' must have a "
                "'save()' method. AutoGluon models typically have this method."
            )
        )

    path = Path(path).resolve()
    path.mkdir(parents=True, exist_ok=True)

    if mlflow_model is None:
        mlflow_model = Model()

    if conda_env is None:
        conda_env = get_default_conda_env(
            model_type=model_type,
            additional_pip_requirements=extra_pip_requirements,
        )
    elif isinstance(conda_env, str):
        conda_env = json.loads(Path(conda_env).read_text())

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        model_type=model_type,
        autogluon_version=kwargs.get("autogluon_version"),
        predictor_metadata=kwargs.get("predictor_metadata", {}),
    )

    mlflow_model.add_flavor(
        "python_function",
        loader_module="mlflow_autogluon.pyfunc",
        model_type=model_type,
    )

    mlflow_model_file_path = path / MLMODEL_FILE_NAME
    mlflow_model.save(mlflow_model_file_path)

    autogluon_model_path = path / AUTODEPLOY_SUBPATH

    if model_type == "tabular":
        temp_save_path = path / "temp_autogluon_save"
        temp_save_path.mkdir(parents=True, exist_ok=True)

        original_path = getattr(autogluon_model, "path", None)
        autogluon_model.save(str(temp_save_path))

        if autogluon_model_path.exists():
            shutil.rmtree(autogluon_model_path)
        shutil.move(str(temp_save_path), str(autogluon_model_path))

        if original_path:
            autogluon_model.path = original_path
    else:
        autogluon_model.save(str(autogluon_model_path))

    metadata = {
        "model_type": model_type,
        "model_class": type(autogluon_model).__name__,
        "model_module": type(autogluon_model).__module__,
    }

    if model_type == "tabular" and hasattr(autogluon_model, "predict"):
        metadata["supports_predict_proba"] = hasattr(autogluon_model, "predict_proba")

    metadata_file_path = path / AUTODEPLOY_METADATA_FILE
    with open(metadata_file_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
