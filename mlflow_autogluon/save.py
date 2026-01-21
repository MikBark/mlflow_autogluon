"""Model saving functionality for AutoGluon MLflow flavor."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.models.model import MLMODEL_FILE_NAME

from mlflow_autogluon.constants import (
    AUTODEPLOY_METADATA_FILE,
    AUTODEPLOY_SUBPATH,
    FLAVOR_NAME,
)
from mlflow_autogluon.domain.model_type import ModelType
from mlflow_autogluon.domain.save_config import SaveConfig
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
    config = SaveConfig.create(
        model_type=model_type,
        conda_env=conda_env,
        pip_requirements=pip_requirements,
        extra_pip_requirements=extra_pip_requirements,
        autogluon_version=kwargs.get("autogluon_version"),
        predictor_metadata=kwargs.get("predictor_metadata", {}),
    )

    _validate_model(autogluon_model, config.model_type)
    path = _prepare_save_path(path)
    mlflow_model = _configure_mlflow_model(mlflow_model, path, config)

    _save_autogluon_model(autogluon_model, path, config.model_type)
    _write_metadata(autogluon_model, path, config.model_type)


def _validate_model(model: Any | object, model_type: ModelType) -> None:
    """Validate that the model has required capabilities.

    Args:
        model: AutoGluon model instance
        model_type: Type of AutoGluon model

    Raises:
        MlflowException: If model lacks save() method
    """
    if not hasattr(model, "save"):
        raise MlflowException(
            message=(
                f"Model of type '{type(model).__name__}' must have a "
                "'save()' method. AutoGluon models typically have this method."
            )
        )


def _prepare_save_path(path: str) -> Path:
    """Prepare and create the save directory.

    Args:
        path: Path where model should be saved

    Returns:
        Resolved Path object
    """
    path_obj = Path(path).resolve()
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def _configure_mlflow_model(
    mlflow_model: Model | None,
    path: Path,
    config: SaveConfig,
) -> Model:
    """Configure MLflow model with flavors and save MLmodel file.

    Args:
        mlflow_model: Existing MLflow model or None
        path: Path where model is being saved
        config: Save configuration

    Returns:
        Configured MLflow Model instance
    """
    if mlflow_model is None:
        mlflow_model = Model()

    conda_env = config.conda_env
    if conda_env is None:
        conda_env = get_default_conda_env(
            model_type=config.model_type.value,
            additional_pip_requirements=config.extra_pip_requirements,
        )
    elif isinstance(conda_env, str):
        conda_env = json.loads(Path(conda_env).read_text())

    mlflow_model.add_flavor(
        FLAVOR_NAME,
        model_type=config.model_type.value,
        autogluon_version=config.autogluon_version,
        predictor_metadata=config.predictor_metadata,
    )

    mlflow_model.add_flavor(
        "python_function",
        loader_module="mlflow_autogluon.pyfunc",
        model_type=config.model_type.value,
    )

    mlflow_model_file_path = path / MLMODEL_FILE_NAME
    mlflow_model.save(mlflow_model_file_path)

    return mlflow_model


def _save_autogluon_model(
    autogluon_model: Any | object,
    path: Path,
    model_type: ModelType,
) -> None:
    """Save the AutoGluon model to the specified path.

    Args:
        autogluon_model: AutoGluon model instance
        path: Path where model should be saved
        model_type: Type of AutoGluon model
    """
    autogluon_model_path = path / AUTODEPLOY_SUBPATH

    if model_type == ModelType.TABULAR:
        _save_tabular_model(autogluon_model, autogluon_model_path)
    else:
        autogluon_model.save(str(autogluon_model_path))


def _save_tabular_model(
    autogluon_model: Any | object,
    autogluon_model_path: Path,
) -> None:
    """Save tabular model with path preservation.

    Tabular models have special handling to preserve the original path
    attribute after saving.

    Args:
        autogluon_model: TabularPredictor instance
        autogluon_model_path: Path where model should be saved
    """
    temp_save_path = autogluon_model_path.parent / "temp_autogluon_save"
    temp_save_path.mkdir(parents=True, exist_ok=True)

    original_path = getattr(autogluon_model, "path", None)
    autogluon_model.save(str(temp_save_path))

    if autogluon_model_path.exists():
        shutil.rmtree(autogluon_model_path)
    shutil.move(str(temp_save_path), str(autogluon_model_path))

    if original_path:
        autogluon_model.path = original_path


def _write_metadata(
    autogluon_model: Any | object,
    path: Path,
    model_type: ModelType,
) -> None:
    """Write model metadata to JSON file.

    Args:
        autogluon_model: AutoGluon model instance
        path: Path where model is saved
        model_type: Type of AutoGluon model
    """
    metadata = {
        "model_type": model_type.value,
        "model_class": type(autogluon_model).__name__,
        "model_module": type(autogluon_model).__module__,
    }

    if model_type == ModelType.TABULAR and hasattr(autogluon_model, "predict"):
        metadata["supports_predict_proba"] = hasattr(autogluon_model, "predict_proba")

    metadata_file_path = path / AUTODEPLOY_METADATA_FILE
    with open(metadata_file_path, "w") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)
