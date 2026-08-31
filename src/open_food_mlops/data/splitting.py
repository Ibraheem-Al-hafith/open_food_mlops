"""Dataset splitting utilities for machine learning pipelines.

This module provides clean, high-level abstractions around dataset splitting,
handling hold-out test set generation and cross-validation/holdout validation
strategies using scikit-learn splitters under the hood.

The primary entry point is ``DatasetSplits``:
    - dataset.splits -> yields (X_train, y_train, X_validation, y_validation)
    - dataset.X_test / y_test -> holds the untouched test partition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    KFold,
    ShuffleSplit,
    StratifiedKFold,
    StratifiedShuffleSplit,
    train_test_split,
)


class ValidationMethod(StrEnum):
    """Supported validation strategies."""

    TRAIN_TEST_SPLIT = "train_test_split"
    K_FOLD = "kfold"
    STRATIFIED_K_FOLD = "stratified_kfold"


@dataclass(frozen=True, slots=True)
class TestConfig:
    """Configuration for the final hold-out test set partition.

    Attributes:
        test_size: Ratio (float) or absolute count (int) reserved for testing.
        stratify: Whether to preserve target class proportions.
        random_state: Random seed for reproducibility.
    """

    test_size: float | int = 0.2
    stratify: bool = True
    random_state: int | None = 42

    def __post_init__(self) -> None:
        """Validate test split configuration parameters."""
        if isinstance(self.test_size, float) and not (0.0 < self.test_size < 1.0):
            raise ValueError("test_size as float must be in the open interval (0.0, 1.0).")
        if isinstance(self.test_size, int) and self.test_size <= 0:
            raise ValueError("test_size as int must be greater than zero.")
        if not isinstance(self.test_size, (float, int)):
            raise TypeError("test_size must be a float or integer.")


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Configuration for validation splitting strategy.

    Attributes:
        method: Validation strategy type.
        n_splits: Number of folds (used for K-Fold / Stratified K-Fold).
        validation_size: Fraction or count for holdout (used for train_test_split).
        shuffle: Whether to shuffle data before splitting.
        stratify: Whether to stratify target classes in validation.
        random_state: Random seed for reproducibility.
    """

    method: ValidationMethod = ValidationMethod.TRAIN_TEST_SPLIT
    n_splits: int = 5
    validation_size: float | int = 0.2
    shuffle: bool = True
    stratify: bool = True
    random_state: int | None = 42

    def __post_init__(self) -> None:
        """Validate validation settings."""
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2.")
        if not self.shuffle and self.random_state is not None:
            raise ValueError("random_state must be None when shuffle=False.")


@dataclass(frozen=True, slots=True)
class DataSplitConfig:
    """Master configuration for test and validation dataset splitting.

    Attributes:
        test: Configuration for test partition.
        validation: Configuration for validation partition(s).
    """

    test: TestConfig = field(default_factory=TestConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Container representing a single train and validation partition pair.

    Attributes:
        X_train: Training features.
        y_train: Training targets.
        X_validation: Validation features.
        y_validation: Validation targets.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series


@dataclass
class DatasetSplits:
    """Holds development and final test partitions of a dataset.

    Provides a clean iterator interface for cross-validation or holdout validation
    while isolating the final holdout test set.

    Attributes:
        X_train: Development feature set.
        y_train: Development target set.
        X_test: Final test feature set.
        y_test: Final test target set.
        config: Full dataset splitting configuration.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    config: DataSplitConfig

    @classmethod
    def from_dataframe(
        cls,
        dataframe: pd.DataFrame,
        target: str,
        config: DataSplitConfig | None = None,
    ) -> DatasetSplits:
        """Construct dataset partitions from a Pandas DataFrame.

        Args:
            dataframe: Complete dataset.
            target: Name of the target column.
            config: Splitting configuration. Defaults to default DataSplitConfig.

        Returns:
            DatasetSplits instance initialized with test and dev partitions.

        Raises:
            TypeError: If dataframe is not a pandas DataFrame.
            ValueError: If dataset is empty, target is missing, or target has NaNs.
        """
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame.")
        if dataframe.empty:
            raise ValueError("Cannot split an empty DataFrame.")
        if target not in dataframe.columns:
            raise ValueError(f"Target column '{target}' non-existent in DataFrame.")
        if dataframe[target].isna().any():
            raise ValueError(f"Target column '{target}' contains missing values.")

        config = config or DataSplitConfig()

        X = dataframe.drop(columns=[target])
        y = dataframe[target]

        stratify = y if config.test.stratify else None
        train_idx, test_idx = train_test_split(
            np.arange(len(dataframe)),
            test_size=config.test.test_size,
            random_state=config.test.random_state,
            stratify=stratify,
        )

        return cls(
            X_train=X.iloc[train_idx].reset_index(drop=True),
            y_train=y.iloc[train_idx].reset_index(drop=True),
            X_test=X.iloc[test_idx].reset_index(drop=True),
            y_test=y.iloc[test_idx].reset_index(drop=True),
            config=config,
        )

    def _get_splitter(self) -> Iterator[tuple[np.ndarray, np.ndarray]]:
        """Instantiate and execute the scikit-learn splitter based on config."""
        v_cfg = self.config.validation

        if v_cfg.method == ValidationMethod.TRAIN_TEST_SPLIT:
            if v_cfg.stratify:
                splitter = StratifiedShuffleSplit(
                    n_splits=1,
                    test_size=v_cfg.validation_size,
                    random_state=v_cfg.random_state,
                )
            else:
                splitter = ShuffleSplit(
                    n_splits=1,
                    test_size=v_cfg.validation_size,
                    random_state=v_cfg.random_state,
                )
            return splitter.split(self.X_train, self.y_train)

        if v_cfg.method == ValidationMethod.K_FOLD:
            splitter = KFold(
                n_splits=v_cfg.n_splits,
                shuffle=v_cfg.shuffle,
                random_state=v_cfg.random_state,
            )
            return splitter.split(self.X_train, self.y_train)

        if v_cfg.method == ValidationMethod.STRATIFIED_K_FOLD:
            splitter = StratifiedKFold(
                n_splits=v_cfg.n_splits,
                shuffle=v_cfg.shuffle,
                random_state=v_cfg.random_state,
            )
            return splitter.split(self.X_train, self.y_train)

        raise ValueError(f"Unsupported validation method: {v_cfg.method!r}")

    @property
    def splits(self) -> Iterator[DatasetSplit]:
        """Lazily yield DatasetSplit pairs for model training and validation."""
        for train_idx, val_idx in self._get_splitter():
            yield DatasetSplit(
                X_train=self.X_train.iloc[train_idx].reset_index(drop=True),
                y_train=self.y_train.iloc[train_idx].reset_index(drop=True),
                X_validation=self.X_train.iloc[val_idx].reset_index(drop=True),
                y_validation=self.y_train.iloc[val_idx].reset_index(drop=True),
            )

    @property
    def n_splits(self) -> int:
        """Return total number of validation iterations/folds."""
        if self.config.validation.method == ValidationMethod.TRAIN_TEST_SPLIT:
            return 1
        return self.config.validation.n_splits

    @property
    def test_size(self) -> int:
        """Return total count of samples in final test set."""
        return len(self.X_test)