"""Core MLflow flavor implementation for AutoGluon models.

This module provides the save/log/load functionality for AutoGluon models
in MLflow, supporting tabular, multimodal, vision, and timeseries predictors.
"""

from __future__ import annotations

from mlflow_autogluon._constants import FLAVOR_NAME
from mlflow_autogluon._load import load_model
from mlflow_autogluon._log import log_model
from mlflow_autogluon._save import save_model

__all__ = [
    "FLAVOR_NAME",
    "save_model",
    "log_model",
    "load_model",
]
