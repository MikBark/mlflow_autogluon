"""Prediction method enum for AutoGluon PyFunc wrapper."""

from __future__ import annotations

from enum import Enum
from typing import Any


class PredictMethod(str, Enum):
    """Supported prediction methods for AutoGluon models."""

    PREDICT = "predict"
    PREDICT_PROBA = "predict_proba"
    PREDICT_MULTI = "predict_multi"

    @classmethod
    def from_string(cls, value: str) -> PredictMethod:
        """Convert string to PredictMethod enum.

        Args:
            value: String predict method (e.g., 'predict', 'predict_proba')

        Returns:
            PredictMethod enum value

        Raises:
            ValueError: If predict_method is not supported
        """
        try:
            return cls(value)
        except ValueError:
            valid = [m.value for m in cls]
            raise ValueError(
                f"Invalid predict_method '{value}'. Supported: {valid}"
            ) from None

    def requires_capability(self) -> str | None:
        """Return the capability name required for this method.

        Returns:
            Capability name (e.g., 'predict_proba') or None if no special
            capability needed
        """
        return _REQUIRED_CAPABILITIES.get(self)

    def validate_capability(self, model: Any) -> None:
        """Validate that the model has the required capability for this method.

        Args:
            model: AutoGluon model instance

        Raises:
            ValueError: If model lacks the required capability
        """
        capability = self.requires_capability()
        if capability and not hasattr(model, capability):
            raise ValueError(
                f"Model {type(model).__name__} does not support {capability}"
            )


_REQUIRED_CAPABILITIES: dict[PredictMethod, str | None] = {
    PredictMethod.PREDICT: None,
    PredictMethod.PREDICT_PROBA: "predict_proba",
    PredictMethod.PREDICT_MULTI: "predict_multi",
}
