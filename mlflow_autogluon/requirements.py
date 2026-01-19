"""Dependency and environment configuration for AutoGluon MLflow flavor."""

from typing import Any

from mlflow_autogluon.mlflow_utils import _mlflow_conda_env


def get_default_pip_requirements(
    model_type: str = "tabular",
) -> list[str]:
    """
    Return default pip requirements for MLflow Models produced by this flavor.

    Args:
        model_type: Type of AutoGluon model
            ('tabular', 'multimodal', 'vision', 'timeseries')

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
