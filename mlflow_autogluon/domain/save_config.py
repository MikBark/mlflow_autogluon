"""Save configuration dataclass for AutoGluon models."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from mlflow_autogluon.domain.model_type import ModelType


@dataclass
class SaveConfig:
    """Configuration for saving AutoGluon models to MLflow.

    Attributes:
        model_type: Type of AutoGluon model
        conda_env: Conda environment dict or path to conda env yaml file
        pip_requirements: Override default pip requirements
        extra_pip_requirements: Extra pip requirements to add to defaults
        autogluon_version: AutoGluon version string
        predictor_metadata: Additional metadata about the predictor
    """

    model_type: ModelType = ModelType.TABULAR
    conda_env: dict[str, Any] | str | None = None
    pip_requirements: list[str] | None = None
    extra_pip_requirements: list[str] | None = None
    autogluon_version: str | None = None
    predictor_metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(cls, model_type: str | ModelType = "tabular", **kwargs: Any) -> SaveConfig:
        """Create a SaveConfig from a string model_type and kwargs.

        Args:
            model_type: String or ModelType enum
            **kwargs: Additional configuration parameters

        Returns:
            SaveConfig instance
        """
        if isinstance(model_type, str):
            model_type = ModelType.from_string(model_type)

        return cls(model_type=model_type, **kwargs)
