"""MLflow plugin for AutoGluon models.

This package provides MLflow integration for AutoGluon models, enabling:
- Model tracking and versioning
- Model Registry support
- Deployment via MLflow serving
- PyFunc compatibility for standardized inference
"""

from mlflow_autogluon.constants import FLAVOR_NAME
from mlflow_autogluon.domain import ModelType, PredictMethod, SaveConfig
from mlflow_autogluon.load import load_model
from mlflow_autogluon.log import log_model
from mlflow_autogluon.requirements import (
    get_default_conda_env,
    get_default_pip_requirements,
)
from mlflow_autogluon.save import save_model

__all__ = [
    "FLAVOR_NAME",
    "ModelType",
    "PredictMethod",
    "SaveConfig",
    "save_model",
    "log_model",
    "load_model",
    "get_default_conda_env",
    "get_default_pip_requirements",
]
