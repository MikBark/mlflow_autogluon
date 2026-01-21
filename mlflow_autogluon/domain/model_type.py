"""Model type enum and mappings for AutoGluon."""

from __future__ import annotations

from enum import Enum


class ModelType(str, Enum):
    """Supported AutoGluon model types."""

    TABULAR = "tabular"
    MULTIMODAL = "multimodal"
    VISION = "vision"
    TIMESERIES = "timeseries"

    @classmethod
    def from_string(cls, value: str) -> ModelType:
        """Convert string to ModelType enum.

        Args:
            value: String model type (e.g., 'tabular', 'multimodal')

        Returns:
            ModelType enum value

        Raises:
            ValueError: If model_type is not supported
        """
        try:
            return cls(value)
        except ValueError:
            valid = [m.value for m in cls]
            raise ValueError(
                f"Unsupported model_type '{value}'. Supported: {valid}"
            ) from None

    @property
    def predictor_class_name(self) -> str:
        """Get the predictor class name for this model type."""
        return _PREDICTOR_CLASS_NAMES[self]

    @property
    def module_path(self) -> str:
        """Get the autogluon module path for this model type."""
        return _MODULE_PATHS[self]

    @property
    def package_name(self) -> str:
        """Get the pip package name for this model type."""
        return _PACKAGE_NAMES[self]


_MODULE_PATHS: dict[ModelType, str] = {
    ModelType.TABULAR: "autogluon.tabular",
    ModelType.MULTIMODAL: "autogluon.multimodal",
    ModelType.VISION: "autogluon.vision",
    ModelType.TIMESERIES: "autogluon.timeseries",
}

_PREDICTOR_CLASS_NAMES: dict[ModelType, str] = {
    ModelType.TABULAR: "TabularPredictor",
    ModelType.MULTIMODAL: "MultiModalPredictor",
    ModelType.VISION: "VisionPredictor",
    ModelType.TIMESERIES: "TimeSeriesPredictor",
}

_PACKAGE_NAMES: dict[ModelType, str] = {
    ModelType.TABULAR: "autogluon.tabular",
    ModelType.MULTIMODAL: "autogluon.multimodal",
    ModelType.VISION: "autogluon.vision",
    ModelType.TIMESERIES: "autogluon.timeseries",
}
