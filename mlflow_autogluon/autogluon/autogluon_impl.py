"""
MLflow integration for AutoGluon TabularPredictor.

This module provides the core MLflow flavor implementation for AutoGluon models.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Any, Optional, Union

from mlflow.artifacts import download_artifacts
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME
try:
    from mlflow.models import ModelInfo
except ImportError:
    from mlflow.models.utils import ModelInfo
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.environment import _mlflow_conda_env

FLAVOR_NAME = "autogluon"

AUTODEPLOY_SUBPATH = "model"
AUTODEPLOY_METADATA_FILE = "autogluon_metadata.json"


def get_default_pip_requirements(
    model_type: str = "tabular",
) -> list[str]:
    """
    Return default pip requirements for MLflow Models produced by this flavor.

    Args:
        model_type: Type of AutoGluon model ('tabular', 'multimodal', 'vision', 'timeseries')

    Returns:
        List of pip requirement strings
    """
    requirements = ["autogluon"]

    if model_type == "tabular":
        requirements.append("autogluon.tabular")
    elif model_type == "multimodal":
        requirements.append("autogluon.multimodal")
    elif model_type == "vision":
        requirements.append("autogluon.vision")
    elif model_type == "timeseries":
        requirements.append("autogluon.timeseries")

    return requirements


def get_default_conda_env(
    model_type: str = "tabular",
    additional_pip_requirements: Optional[list[str]] = None,
) -> dict[str, Any]:
    """
    Return default Conda environment for MLflow Models produced by this flavor.

    Args:
        model_type: Type of AutoGluon model
        additional_pip_requirements: Additional pip requirements to include

    Returns:
        Dictionary representing Conda environment specification
    """
    pip_requirements = get_default_pip_requirements(model_type)

    if additional_pip_requirements:
        pip_requirements = list(set(pip_requirements + additional_pip_requirements))

    return _mlflow_conda_env(
        additional_pip_requirements=pip_requirements,
        install_mlflow=False,
    )


def save_model(
    autogluon_model: Union[Any, object],
    path: str,
    model_type: str = "tabular",
    mlflow_model: Optional[Model] = None,
    conda_env: Optional[Union[dict, str]] = None,
    pip_requirements: Optional[list[str]] = None,
    extra_pip_requirements: Optional[list[str]] = None,
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
    supported_types = ["tabular", "multimodal", "vision", "timeseries"]
    if model_type not in supported_types:
        raise MlflowException(
            invalid_parameter_value=INVALID_PARAMETER_VALUE,
            message=f"Unsupported model_type '{model_type}'. Supported types: {supported_types}",
        )

    if not hasattr(autogluon_model, "save"):
        raise MlflowException(
            message=f"Model of type '{type(autogluon_model).__name__}' must have a 'save()' method. "
            f"AutoGluon models typically have this method."
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

    from mlflow_autogluon.autogluon.pyfunc.autogluon_pyfunc import (
        _AutoGluonModelWrapper,
    )

    mlflow_model.add_flavor(
        "python_function",
        loader_module="mlflow_autogluon.autogluon.pyfunc",
        model_type=model_type,
        python_model=_AutoGluonModelWrapper,
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
    with open(metadata_file_path, "w") as f:
        json.dump(metadata, f, indent=2)


def log_model(
    autogluon_model: Union[Any, object],
    artifact_path: str,
    model_type: str = "tabular",
    conda_env: Optional[Union[dict, str]] = None,
    pip_requirements: Optional[list[str]] = None,
    extra_pip_requirements: Optional[list[str]] = None,
    registered_model_name: Optional[str] = None,
    signature: Optional[Any] = None,
    input_example: Optional[Any] = None,
    **kwargs: Any,
) -> ModelInfo:
    """
    Log an AutoGluon model as an MLflow artifact for the current run.

    Args:
        autogluon_model: AutoGluon model instance to log
        artifact_path: Artifact path (relative to run's artifact root)
        model_type: Type of AutoGluon model
        conda_env: Conda environment dict or path
        pip_requirements: Override default pip requirements
        extra_pip_requirements: Extra pip requirements to add
        registered_model_name: Name to register model in Model Registry
        signature: Model signature for input/output schema
        input_example: Example input for model inference
        **kwargs: Additional arguments passed to save_model

    Returns:
        ModelInfo: Logged model info including URI
    """
    return Model.log(
        artifact_path=artifact_path,
        flavor=mlflow_autogluon.autogluon.autogluon_impl,
        autogluon_model=autogluon_model,
        model_type=model_type,
        conda_env=conda_env,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        **kwargs,
    )


def load_model(
    model_uri: str,
    dst_path: Optional[str] = None,
) -> Union[Any, object]:
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
    from mlflow_autogluon.autogluon.pyfunc.autogluon_pyfunc import (
        _AutoGluonModelWrapper,
    )

    return _AutoGluonModelWrapper(path)
