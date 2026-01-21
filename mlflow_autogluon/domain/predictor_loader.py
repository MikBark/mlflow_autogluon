"""Predictor loading strategies for AutoGluon models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from mlflow_autogluon.domain.model_type import ModelType


class PredictorLoader(ABC):
    """Abstract base for predictor loading strategies."""

    @abstractmethod
    def load(self, model_path: str) -> Any:
        """Load an AutoGluon predictor from the given path.

        Args:
            model_path: Path to the saved model

        Returns:
            Loaded AutoGluon predictor instance
        """


class TabularPredictorLoader(PredictorLoader):
    """Loader for tabular models."""

    def load(self, model_path: str) -> Any:
        from autogluon.tabular import TabularPredictor

        return TabularPredictor.load(model_path)


class MultiModalPredictorLoader(PredictorLoader):
    """Loader for multimodal models."""

    def load(self, model_path: str) -> Any:
        from autogluon.multimodal import MultiModalPredictor

        return MultiModalPredictor.load(model_path)


class VisionPredictorLoader(PredictorLoader):
    """Loader for vision models."""

    def load(self, model_path: str) -> Any:
        from autogluon.vision import VisionPredictor

        return VisionPredictor.load(model_path)


class TimeSeriesPredictorLoader(PredictorLoader):
    """Loader for timeseries models."""

    def load(self, model_path: str) -> Any:
        from autogluon.timeseries import TimeSeriesPredictor

        return TimeSeriesPredictor.load(model_path)


_LOADER_REGISTRY: dict[ModelType, PredictorLoader] = {
    ModelType.TABULAR: TabularPredictorLoader(),
    ModelType.MULTIMODAL: MultiModalPredictorLoader(),
    ModelType.VISION: VisionPredictorLoader(),
    ModelType.TIMESERIES: TimeSeriesPredictorLoader(),
}


def get_loader(model_type: ModelType) -> PredictorLoader:
    """Get the appropriate loader for the given model type.

    Args:
        model_type: Type of AutoGluon model

    Returns:
        PredictorLoader instance for the model type

    Raises:
        ValueError: If model_type is not supported
    """
    if model_type not in _LOADER_REGISTRY:
        valid = [t.value for t in ModelType]
        raise ValueError(f"Unsupported model_type '{model_type}'. Supported: {valid}")
    return _LOADER_REGISTRY[model_type]
