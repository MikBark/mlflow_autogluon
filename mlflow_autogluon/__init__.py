"""MLflow plugin for AutoGluon models.

This package provides MLflow integration for AutoGluon models, enabling:
- Model tracking and versioning
- Model Registry support
- Deployment via MLflow serving
- PyFunc compatibility for standardized inference
"""

from mlflow_autogluon.autogluon.autogluon_impl import (
    FLAVOR_NAME,
    get_default_conda_env,
    get_default_pip_requirements,
    load_model,
    log_model,
    save_model,
)

__all__ = [
    "FLAVOR_NAME",
    "save_model",
    "log_model",
    "load_model",
    "get_default_conda_env",
    "get_default_pip_requirements",
]
