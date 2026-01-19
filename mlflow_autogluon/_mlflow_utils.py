"""MLflow-specific utilities and constants for AutoGluon flavor."""

from mlflow.models.model import MLMODEL_FILE_NAME
from mlflow.protos.databricks_pb2 import INVALID_PARAMETER_VALUE
from mlflow.utils.environment import _mlflow_conda_env

MLMODEL_FILE_NAME = MLMODEL_FILE_NAME
INVALID_PARAMETER_VALUE = INVALID_PARAMETER_VALUE
_mlflow_conda_env = _mlflow_conda_env
