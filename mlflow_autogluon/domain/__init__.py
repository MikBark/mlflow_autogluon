"""Domain objects for AutoGluon MLflow flavor."""

from mlflow_autogluon.domain.model_type import ModelType
from mlflow_autogluon.domain.predict_method import PredictMethod
from mlflow_autogluon.domain.save_config import SaveConfig

__all__ = ["ModelType", "PredictMethod", "SaveConfig"]
