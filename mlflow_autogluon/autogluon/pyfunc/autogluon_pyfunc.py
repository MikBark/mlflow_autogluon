"""PyFunc wrapper for AutoGluon models.

This module provides a PythonModel-based wrapper that enables AutoGluon models
to be used with mlflow.pyfunc.load_model() for standardized inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Union

import pandas as pd
from mlflow.pyfunc import PythonModel


class _AutoGluonModelWrapper(PythonModel):
    """
    PyFunc wrapper for AutoGluon models.

    This wrapper enables AutoGluon models to be loaded via mlflow.pyfunc.load_model()
    and provides a standardized predict() interface that accepts pandas DataFrames.
    """

    def __init__(self, path: Union[str, Path] = None, autogluon_model: Any = None):
        """
        Initialize the wrapper.

        Args:
            path: Path to saved model directory
            autogluon_model: Pre-loaded AutoGluon model instance
        """
        if autogluon_model is not None:
            self._model = autogluon_model
        elif path is not None:
            self._model_path = path
            self._model = None
        else:
            raise ValueError("Either path or autogluon_model must be provided")

    def load_context(self, context: Any) -> None:
        """
        Load the AutoGluon model from the artifact path.

        This method is called by MLflow before predict().

        Args:
            context: MLflow context containing artifact path
        """
        if self._model is None:
            from mlflow_autogluon.autogluon.autogluon_impl import load_model

            model_path = getattr(context, "artifacts", self._model_path)
            self._model = load_model(model_path)

    def predict(
        self,
        context: Any,
        model_input: Union[pd.DataFrame, dict[str, Any]],
        params: Optional[dict[str, Any]] = None,
    ) -> Union[pd.DataFrame, dict[str, Any], list[Any]]:
        """
        Generate predictions using the AutoGluon model.

        Args:
            context: MLflow context
            model_input: Input data as pandas DataFrame or dict
            params: Optional prediction parameters:
                - 'predict_method': 'predict' (default) or 'predict_proba'
                - 'as_multiclass': For binary classification, return both class probs
                - 'as_pandas': Return pandas DataFrame (default) or dict/list

        Returns:
            Predictions as DataFrame, dict, or list depending on params

        Raises:
            ValueError: If predict_method is invalid or not supported
        """
        if self._model is None:
            self.load_context(context)

        if params is None:
            params = {}

        predict_method = params.get("predict_method", "predict")

        if isinstance(model_input, dict):
            if "dataframe_split" in model_input:
                model_input = pd.DataFrame(**model_input["dataframe_split"])
            elif "dataframe_records" in model_input:
                model_input = pd.DataFrame(model_input["dataframe_records"])
            else:
                model_input = pd.DataFrame(model_input)
        elif not isinstance(model_input, pd.DataFrame):
            model_input = pd.DataFrame(model_input)

        if predict_method == "predict":
            result = self._model.predict(model_input)
        elif predict_method == "predict_proba":
            if not hasattr(self._model, "predict_proba"):
                raise ValueError(
                    f"Model {type(self._model).__name__} does not support predict_proba"
                )
            as_multiclass = params.get("as_multiclass", False)
            result = self._model.predict_proba(model_input, as_multiclass=as_multiclass)
        elif predict_method == "predict_multi":
            if not hasattr(self._model, "predict_multi"):
                raise ValueError(
                    f"Model {type(self._model).__name__} does not support predict_multi"
                )
            result = self._model.predict_multi(model_input)
        else:
            raise ValueError(
                f"Invalid predict_method '{predict_method}'. "
                f"Supported: 'predict', 'predict_proba', 'predict_multi'"
            )

        as_pandas = params.get("as_pandas", True)
        if not as_pandas and isinstance(result, pd.DataFrame):
            return result.to_dict(orient="records")

        return result
