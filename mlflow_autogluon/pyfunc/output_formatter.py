"""Output formatting utilities for PyFunc wrapper."""

from __future__ import annotations

from typing import Any

import pandas as pd


def format_output(
    output: pd.DataFrame | dict[str, Any] | list[Any],
    as_pandas: bool,
) -> pd.DataFrame | dict[str, Any] | list[Any]:
    """Format prediction output based on requested format.

    Args:
        output: Raw prediction output
        as_pandas: Whether to return pandas DataFrame (True) or convert to
            dict/list (False)

    Returns:
        Formatted output as DataFrame, dict, or list
    """
    if not as_pandas and isinstance(output, pd.DataFrame):
        return output.to_dict(orient="records")
    return output
