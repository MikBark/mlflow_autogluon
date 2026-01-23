"""Type literals and utilities for AutoGluon."""

from __future__ import annotations

from typing import Literal

ModelTypeLiteral = Literal['tabular', 'multimodal', 'vision', 'timeseries']
PredictMethodLiteral = Literal['predict', 'predict_proba', 'predict_multi']
