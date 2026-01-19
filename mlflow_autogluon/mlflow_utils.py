"""Lazy-loaded MLflow imports to reduce top-level import count."""

from mlflow.models.model import MLMODEL_FILE_NAME, ModelInfo
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.environment import _mlflow_conda_env

MLMODEL_FILE_NAME = MLMODEL_FILE_NAME
INVALID_PARAMETER_VALUE = INVALID_PARAMETER_VALUE
ModelInfo = ModelInfo
_mlflow_conda_env = _mlflow_conda_env
