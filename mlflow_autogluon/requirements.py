"""Dependency and environment configuration for AutoGluon MLflow flavor."""

from __future__ import annotations

from typing import Any

from mlflow.utils.environment import _mlflow_conda_env

from mlflow_autogluon.domain.model_type import ModelType


def get_default_pip_requirements(
    model_type: str | ModelType = "tabular",
) -> list[str]:
    """
    Return default pip requirements for MLflow Models produced by this flavor.

    Args:
        model_type: Type of AutoGluon model
            ('tabular', 'multimodal', 'vision', 'timeseries')

    Returns:
        List of pip requirement strings
    """
    if isinstance(model_type, str):
        model_type = ModelType.from_string(model_type)

    requirements = ["autogluon", model_type.package_name]
    return requirements


def get_default_conda_env(
    model_type: str | ModelType = "tabular",
    additional_pip_requirements: list[str] | None = None,
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
        additional_pip_deps=pip_requirements,
        install_mlflow=False,
    )
