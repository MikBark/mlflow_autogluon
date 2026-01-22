"""Type literals and utilities for AutoGluon."""

from __future__ import annotations

from typing import Any, Literal

ModelTypeLiteral = Literal['tabular', 'multimodal', 'vision', 'timeseries']
PredictMethodLiteral = Literal['predict', 'predict_proba', 'predict_multi']


def get_model_loader(model_type: ModelTypeLiteral) -> Any:
    """Get the loader function for the given model type.

    Args:
        model_type: Type of AutoGluon model

    Returns:
        Loader function that takes a path and returns a loaded model

    Raises:
        ValueError: If model_type is not supported
    """
    if model_type == 'tabular':
        from autogluon.tabular import TabularPredictor

        return TabularPredictor.load
    if model_type == 'multimodal':
        from autogluon.multimodal import MultiModalPredictor

        return MultiModalPredictor.load
    if model_type == 'vision':
        from autogluon.vision import VisionPredictor

        return VisionPredictor.load
    if model_type == 'timeseries':
        from autogluon.timeseries import TimeSeriesPredictor

        return TimeSeriesPredictor.load

    raise ValueError(
        f"Unsupported model_type '{model_type}'. "
        f"Supported: ['tabular', 'multimodal', 'vision', 'timeseries']",
    )
