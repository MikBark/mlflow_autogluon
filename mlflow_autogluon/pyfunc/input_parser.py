"""Input parsing strategies for PyFunc wrapper."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class InputParser(ABC):
    """Abstract base for input parsing strategies."""

    @abstractmethod
    def can_parse(self, model_input: Any) -> bool:
        """Check if this parser can handle the given input.

        Args:
            model_input: Input data to check

        Returns:
            True if this parser can handle the input
        """

    @abstractmethod
    def parse(self, model_input: Any) -> pd.DataFrame:
        """Parse the input into a pandas DataFrame.

        Args:
            model_input: Input data to parse

        Returns:
            Parsed pandas DataFrame
        """


class DataFramePassthroughParser(InputParser):
    """Parser that passes through DataFrames unchanged."""

    def can_parse(self, model_input: Any) -> bool:
        return isinstance(model_input, pd.DataFrame)

    def parse(self, model_input: Any) -> pd.DataFrame:
        return model_input


class DataframeSplitParser(InputParser):
    """Parser for dict with 'dataframe_split' format (MLflow standard)."""

    def can_parse(self, model_input: Any) -> bool:
        return (
            isinstance(model_input, dict)
            and "dataframe_split" in model_input
            and isinstance(model_input["dataframe_split"], dict)
        )

    def parse(self, model_input: Any) -> pd.DataFrame:
        return pd.DataFrame(**model_input["dataframe_split"])


class DataframeRecordsParser(InputParser):
    """Parser for dict with 'dataframe_records' format (MLflow standard)."""

    def can_parse(self, model_input: Any) -> bool:
        return (
            isinstance(model_input, dict)
            and "dataframe_records" in model_input
            and isinstance(model_input["dataframe_records"], list)
        )

    def parse(self, model_input: Any) -> pd.DataFrame:
        return pd.DataFrame(model_input["dataframe_records"])


class DictParser(InputParser):
    """Parser for generic dict input."""

    def can_parse(self, model_input: Any) -> bool:
        return isinstance(model_input, dict)

    def parse(self, model_input: Any) -> pd.DataFrame:
        return pd.DataFrame(model_input)


class GenericParser(InputParser):
    """Fallback parser for any other input type."""

    def can_parse(self, model_input: Any) -> bool:
        return True

    def parse(self, model_input: Any) -> pd.DataFrame:
        return pd.DataFrame(model_input)


_PARSERS: list[InputParser] = [
    DataFramePassthroughParser(),
    DataframeSplitParser(),
    DataframeRecordsParser(),
    DictParser(),
    GenericParser(),
]


def parse_input(model_input: pd.DataFrame | dict[str, Any] | Any) -> pd.DataFrame:
    """Parse input data into a pandas DataFrame.

    Args:
        model_input: Input data as DataFrame, dict, or other format

    Returns:
        Parsed pandas DataFrame
    """
    for parser in _PARSERS:
        if parser.can_parse(model_input):
            return parser.parse(model_input)

    return GenericParser().parse(model_input)
