"""Column selector feature transformer for dataset feature selection and validation."""

from __future__ import annotations

import logging
from collections.abc import Sequence
from typing import Literal

import pandas as pd

from open_food_mlops.features.base import BaseFeatureTransformer

logger = logging.getLogger(__name__)

MissingColumnsStrategy = Literal["error", "warning", "ignore"]


class ColumnSelectorTransformer(BaseFeatureTransformer):
    """Transformer to select a specific subset of columns from a pandas DataFrame.

    Validates presence of requested columns and handles missing columns according
    to the configured `missing_columns` policy.

    Attributes:
        columns: List of column names to select.
        missing_columns: Handling strategy for missing columns ('error', 'warning', 'ignore').
    """

    def __init__(
        self,
        columns: Sequence[str],
        missing_columns: MissingColumnsStrategy = "error",
    ) -> None:
        """Initialize ColumnSelectorTransformer.

        Args:
            columns: Ordered sequence of column names to select.
            missing_columns: Action to take if columns are missing:
                - 'error': Raise KeyError listing missing columns.
                - 'warning': Log a warning with missing columns and return existing ones.
                - 'ignore': Silently filter and return existing columns.

        Raises:
            ValueError: If `columns` is empty or `missing_columns` mode is invalid.
        """
        super().__init__()
        if not columns:
            raise ValueError("The `columns` argument cannot be empty.")

        valid_modes: tuple[MissingColumnsStrategy, ...] = ("error", "warning", "ignore")
        if missing_columns not in valid_modes:
            raise ValueError(
                f"Invalid `missing_columns` mode '{missing_columns}'. "
                f"Must be one of {valid_modes}."
            )

        self.columns: tuple[str, ...] = tuple(columns)
        self.missing_columns: MissingColumnsStrategy = missing_columns

    def _fit_transformer(self, X: pd.DataFrame, y: pd.Series | None) -> None:
        """Validate input columns during fit according to configured strategy."""
        self._validate_missing_columns(X.columns)

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select configured columns present in the input DataFrame."""
        existing_cols = [col for col in self.columns if col in X.columns]

        if self.missing_columns == "error" and len(existing_cols) != len(self.columns):
            self._validate_missing_columns(X.columns)

        return X[existing_cols].copy()

    def _validate_missing_columns(self, dataframe_columns: Sequence[str]) -> None:
        """Check for missing columns and apply the configured strategy."""
        missing = [col for col in self.columns if col not in dataframe_columns]
        if not missing:
            return

        msg = f"The following requested columns are not present in the DataFrame: {missing}"

        if self.missing_columns == "error":
            logger.error(msg)
            raise KeyError(msg)

        if self.missing_columns == "warning":
            logger.warning(msg)