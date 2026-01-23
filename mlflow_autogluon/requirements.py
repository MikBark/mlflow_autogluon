"""Dependency and environment configuration for AutoGluon MLflow flavor."""

from __future__ import annotations

from typing import Any

from mlflow.utils.environment import _mlflow_conda_env

from mlflow_autogluon.constants import MODEL_PACKAGES
from mlflow_autogluon.literals import ModelTypeLiteral


def get_default_pip_requirements(
    model_type: ModelTypeLiteral = 'tabular',
) -> list[str]:
    """
    Return default pip requirements for MLflow Models produced by this flavor.

    Args:
        model_type: Type of AutoGluon model
            ('tabular', 'multimodal', 'vision', 'timeseries')

    Returns:
        List of pip requirement strings
    """
    return ['autogluon', MODEL_PACKAGES[model_type]]


def get_default_conda_env(
    model_type: ModelTypeLiteral = 'tabular',
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
