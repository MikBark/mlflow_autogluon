"""Model logging functionality for AutoGluon MLflow flavor."""

from __future__ import annotations

import sys
from typing import Any

from mlflow.models import Model
from mlflow.models.model import ModelInfo

from mlflow_autogluon.predict_methods import ModelTypeLiteral


def log_model(  # noqa: WPS211,WPS213
    autogluon_model: Any | object,
    artifact_path: str,
    model_type: ModelTypeLiteral = 'tabular',
    conda_env: dict | str | None = None,
    pip_requirements: list[str] | None = None,
    extra_pip_requirements: list[str] | None = None,
    registered_model_name: str | None = None,
    signature: Any | None = None,
    input_example: Any | None = None,
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
        flavor=sys.modules['mlflow_autogluon'],
        autogluon_model=autogluon_model,
        model_type=model_type,
        conda_env=conda_env,
        registered_model_name=registered_model_name,
        signature=signature,
        input_example=input_example,
        **kwargs,
    )
