"""Integration tests for MLflow-AutoGluon."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd


def test_mlflow_plugin_registration():
    """Test that the plugin can be imported."""
    import mlflow_autogluon

    assert hasattr(mlflow_autogluon, 'FLAVOR_NAME')
    assert hasattr(mlflow_autogluon, 'save_model')
    assert hasattr(mlflow_autogluon, 'log_model')
    assert hasattr(mlflow_autogluon, 'load_model')


def test_end_to_end_save_and_load_with_mock_model():
    """Test end-to-end save and load with mock model."""
    from mlflow_autogluon import load_model, save_model

    class MockAutoGluonModel:
        def __init__(self):
            self.path = None

        def save(self, path):
            self.path = path
            model_path = Path(path) / 'model'
            model_path.mkdir(parents=True, exist_ok=True)

        def predict(self, data):
            return pd.Series([1, 2, 3])

    with tempfile.TemporaryDirectory() as tmp:
        model = MockAutoGluonModel()

        save_model(
            autogluon_model=model,
            path=tmp,
            model_type='tabular',
        )

        mlmodel_path = Path(tmp) / 'MLmodel'
        assert mlmodel_path.exists()

        with patch('autogluon.tabular.TabularPredictor') as mock_predictor:
            mock_predictor.load.return_value = model

            loaded_model = load_model(tmp)
            assert loaded_model is not None
            mock_predictor.load.assert_called_once()


@patch('autogluon.tabular.TabularPredictor')
def test_load_model_calls_tabular_predictor_load(mock_predictor):
    """Test that load_model calls TabularPredictor.load()."""

    from mlflow_autogluon import load_model

    with tempfile.TemporaryDirectory() as tmp:
        # Create the necessary directory structure and MLmodel file
        Path(tmp).mkdir(parents=True, exist_ok=True)
        model_path = Path(tmp) / 'model'
        model_path.mkdir(parents=True, exist_ok=True)

        # Create a minimal MLmodel file
        import json

        mlmodel_path = Path(tmp) / 'MLmodel'
        mlmodel_path.write_text(
            json.dumps(
                {
                    'flavors': {
                        'autogluon': {'model_type': 'tabular'},
                    }
                }
            )
        )

        mock_predictor.load.return_value = MagicMock()

        load_model(tmp)

        mock_predictor.load.assert_called_once()
