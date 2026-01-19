"""
Core MLflow flavor tests for AutoGluon integration.
"""

import tempfile
from pathlib import Path

import pytest

from mlflow_autogluon import (
    FLAVOR_NAME,
    get_default_conda_env,
    get_default_pip_requirements,
    save_model,
)


def test_flavor_name():
    """Test FLAVOR_NAME constant."""
    assert FLAVOR_NAME == "autogluon"


def test_get_default_pip_requirements():
    """Test pip requirements generation."""
    reqs = get_default_pip_requirements(model_type="tabular")
    assert "autogluon" in reqs
    assert "autogluon.tabular" in reqs

    reqs = get_default_pip_requirements(model_type="multimodal")
    assert "autogluon.multimodal" in reqs


def test_get_default_conda_env():
    """Test conda environment generation."""
    env = get_default_conda_env(model_type="tabular")
    assert "dependencies" in env
    assert "pip" in str(env)


def test_save_model_invalid_type():
    """Test save_model with invalid model type."""
    from mlflow.exceptions import MlflowException

    class FakeModel:
        def save(self, path):
            pass

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(MlflowException) as exc_info:
            save_model(
                autogluon_model=FakeModel(),
                path=tmp,
                model_type="invalid_type",
            )
        assert "Unsupported model_type" in str(exc_info.value)


def test_save_model_without_save_method():
    """Test save_model with model that lacks save() method."""
    from mlflow.exceptions import MlflowException

    class BadModel:
        pass

    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(MlflowException) as exc_info:
            save_model(
                autogluon_model=BadModel(),
                path=tmp,
                model_type="tabular",
            )
        assert "must have a 'save()' method" in str(exc_info.value)


def test_save_model_creates_mlmodel_file():
    """Test that save_model creates MLmodel file."""

    class FakeModel:
        def save(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        save_model(
            autogluon_model=FakeModel(),
            path=tmp,
            model_type="tabular",
        )

        mlmodel_path = Path(tmp) / "MLmodel"
        assert mlmodel_path.exists()


def test_save_model_creates_model_subdirectory():
    """Test that save_model creates model subdirectory."""

    class FakeModel:
        def save(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        save_model(
            autogluon_model=FakeModel(),
            path=tmp,
            model_type="tabular",
        )

        model_subpath = Path(tmp) / "model"
        assert model_subpath.exists()


def test_log_model_creates_run_if_none_exists():
    """Test that log_model creates a run if none exists (MLflow 3.x behavior)."""
    import mlflow
    from mlflow_autogluon import log_model

    class FakeModel:
        def save(self, path):
            Path(path).mkdir(parents=True, exist_ok=True)

    # Ensure no active run
    mlflow.end_run()

    # In MLflow 3.x, log_model auto-creates a run if none exists
    with tempfile.TemporaryDirectory() as tmp:
        tracking_uri = f"file://{tmp}/mlruns"
        mlflow.set_tracking_uri(tracking_uri)

        model_info = log_model(
            autogluon_model=FakeModel(),
            artifact_path="model",
            model_type="tabular",
        )

        assert model_info is not None
        assert model_info.model_uri.startswith("runs:/")
