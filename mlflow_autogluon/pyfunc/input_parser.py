"""Input parsing utilities for PyFunc wrapper."""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class SupportsDataFrameConversion(Protocol):
    """Protocol for objects that can be converted to DataFrame."""

    def __init__(self, *args: Any, **kwargs: Any) -> None: ...


def parse_input(model_input: pd.DataFrame | dict[str, Any] | Any) -> pd.DataFrame:
    """Parse input data into a pandas DataFrame.

    Args:
        model_input: Input data as DataFrame, dict, or other format

    Returns:
        Parsed pandas DataFrame
    """
    if isinstance(model_input, pd.DataFrame):
        return model_input

    if isinstance(model_input, dict):
        if "dataframe_split" in model_input:
            return pd.DataFrame(**model_input["dataframe_split"])
        if "dataframe_records" in model_input:
            return pd.DataFrame(model_input["dataframe_records"])
        return pd.DataFrame(model_input)

    return pd.DataFrame(model_input)
