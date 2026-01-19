"""
Integration tests for MLflow-AutoGluon.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlflow
import pandas as pd
import pytest


def test_mlflow_plugin_registration():
    """Test that the plugin can be imported."""
    import mlflow_autogluon

    assert hasattr(mlflow_autogluon, "FLAVOR_NAME")
    assert hasattr(mlflow_autogluon, "save_model")
    assert hasattr(mlflow_autogluon, "log_model")
    assert hasattr(mlflow_autogluon, "load_model")


def test_end_to_end_save_and_load_with_mock_model():
    """Test end-to-end save and load with mock model."""
    from mlflow_autogluon import save_model, load_model, FLAVOR_NAME

    class MockAutoGluonModel:
        def __init__(self):
            self.path = None

        def save(self, path):
            self.path = path
            model_path = Path(path) / "model"
            model_path.mkdir(parents=True, exist_ok=True)

        def predict(self, data):
            return pd.Series([1, 2, 3])

    with tempfile.TemporaryDirectory() as tmp:
        model = MockAutoGluonModel()

        save_model(
            autogluon_model=model,
            path=tmp,
            model_type="tabular",
        )

        mlmodel_path = Path(tmp) / "MLmodel"
        assert mlmodel_path.exists()

        with patch("mlflow_autogluon.autogluon.autogluon_impl.TabularPredictor") as mock_predictor:
            mock_predictor.load.return_value = model

            loaded_model = load_model(tmp)
            assert loaded_model is not None
            mock_predictor.load.assert_called_once()


@patch("mlflow_autogluon.autogluon.autogluon_impl.TabularPredictor")
def test_load_model_calls_tabular_predictor_load(mock_predictor):
    """Test that load_model calls TabularPredictor.load()."""
    from mlflow_autogluon import load_model
    from mlflow.tracking.artifact_utils import _download_artifact_from_uri

    with tempfile.TemporaryDirectory() as tmp:
        Path(tmp).mkdir(parents=True, exist_ok=True)
        model_path = Path(tmp) / "model"
        model_path.mkdir(parents=True, exist_ok=True)

        with patch("mlflow_autogluon.autogluon.autogluon_impl._download_artifact_from_uri") as mock_download:
            mock_download.return_value = tmp

            mock_predictor.load.return_value = MagicMock()

            load_model("runs:/123/model")

            mock_predictor.load.assert_called()
