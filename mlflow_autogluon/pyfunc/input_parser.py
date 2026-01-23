"""Input parsing utilities for PyFunc wrapper."""

from __future__ import annotations

from typing import Any

import pandas as pd


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
        if 'dataframe_split' in model_input:
            return pd.DataFrame(**model_input['dataframe_split'])
        if 'dataframe_records' in model_input:
            return pd.DataFrame(model_input['dataframe_records'])
        return pd.DataFrame(model_input)

    return pd.DataFrame(model_input)
