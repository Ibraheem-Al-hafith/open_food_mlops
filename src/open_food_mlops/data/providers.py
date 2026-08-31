"""Data provider abstractions for train-validation-test partitioning."""

from __future__ import annotations

from abc import ABC, abstractmethod
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
    """Supported validation strategy enumeration."""

    TRAIN_TEST_SPLIT = "train_test_split"
    K_FOLD = "kfold"
    STRATIFIED_K_FOLD = "stratified_kfold"


@dataclass(frozen=True, slots=True)
class DataSplit:
    """Encapsulates features and targets for a train/validation split pair."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    fold: int = 0


@dataclass(frozen=True, slots=True)
class HoldoutTestConfig:
    """Configuration for reserving an untouched hold-out test partition.

    Attributes:
        test_size: Ratio (0.0 to 1.0) or exact count reserved for test evaluation.
        stratify: Whether to preserve class proportions in the test split.
        random_state: Seed for reproducible random splitting.
    """

    test_size: float | int = 0.2
    stratify: bool = True
    random_state: int | None = 42

    def __post_init__(self) -> None:
        """Validate hold-out configuration attributes."""
        if isinstance(self.test_size, float) and not (0.0 < self.test_size < 1.0):
            raise ValueError("test_size as float must be in range (0.0, 1.0).")
        if isinstance(self.test_size, int) and self.test_size <= 0:
            raise ValueError("test_size as int must be greater than zero.")


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Configuration for train/validation cross-validation splitting.

    Attributes:
        method: Splitting methodology (KFold, StratifiedKFold, Train/Test).
        n_splits: Number of cross-validation folds.
        validation_size: Holdout size fraction if method is TRAIN_TEST_SPLIT.
        shuffle: Whether to shuffle data samples before splitting.
        stratify: Whether to preserve class balance across folds.
        random_state: Seed for reproducible random splitting.
    """

    method: ValidationMethod = ValidationMethod.STRATIFIED_K_FOLD
    n_splits: int = 5
    validation_size: float | int = 0.2
    shuffle: bool = True
    stratify: bool = True
    random_state: int | None = 42

    def __post_init__(self) -> None:
        """Validate cross-validation configuration parameters."""
        if self.n_splits < 2:
            raise ValueError("n_splits must be at least 2 for validation.")
        if not self.shuffle and self.random_state is not None:
            raise ValueError("random_state must be None when shuffle is False.")


@dataclass(frozen=True, slots=True)
class DataPartitionConfig:
    """Master dataset partitioning specification combining test and validation configs."""

    test: HoldoutTestConfig = field(default_factory=HoldoutTestConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


class BaseDataProvider(ABC):
    """Abstract Base Class for training data provision and split generation."""

    @abstractmethod
    def get_splits(self) -> Iterator[DataSplit]:
        """Yield DataSplit instances across configured folds."""

    @property
    @abstractmethod
    def test_set(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return held-out test feature and target partitions."""


class TabularDataProvider(BaseDataProvider):
    """Production provider yielding standardized train-validation splits and test set.

    Reads processed data files (Parquet or CSV), isolates a strict hold-out test dataset,
    and lazily streams train/validation splits for experiment orchestrations.
    """

    def __init__(
        self,
        data_path: str,
        target_column: str = "nova_group",
        config: DataPartitionConfig | None = None,
    ) -> None:
        """Initialize provider by loading data and creating primary test-dev partitions.

        Args:
            data_path: File path to the processed parquet or csv dataset.
            target_column: Name of the classification or regression target column.
            config: Dataset partition configuration. Defaults to standard settings.

        Raises:
            FileNotFoundError: If dataset path does not exist.
            KeyError: If target_column is missing from dataset.
            ValueError: If dataset is empty or target contains missing values.
        """
        self.data_path = data_path
        self.target_column = target_column
        self.config = config or DataPartitionConfig()

        self._load_and_partition()

    def _load_and_partition(self) -> None:
        """Loads data from disk and isolates holdout test data from training pool."""
        if self.data_path.endswith(".parquet"):
            df = pd.read_parquet(self.data_path)
        else:
            df = pd.read_csv(self.data_path)

        if df.empty:
            raise ValueError("Cannot process empty dataset.")

        if self.target_column not in df.columns:
            raise KeyError(
                f"Target column '{self.target_column}' not found in dataset at {self.data_path}"
            )

        if df[self.target_column].isna().any():
            raise ValueError(
                f"Target column '{self.target_column}' contains unhandled missing values."
            )

        X = df.drop(columns=[self.target_column])
        y = df[self.target_column]

        stratify = y if self.config.test.stratify else None

        train_idx, test_idx = train_test_split(
            np.arange(len(df)),
            test_size=self.config.test.test_size,
            random_state=self.config.test.random_state,
            stratify=stratify,
        )

        self._X_dev = X.iloc[train_idx].reset_index(drop=True)
        self._y_dev = y.iloc[train_idx].reset_index(drop=True)
        self._X_test = X.iloc[test_idx].reset_index(drop=True)
        self._y_test = y.iloc[test_idx].reset_index(drop=True)

    @property
    def test_set(self) -> tuple[pd.DataFrame, pd.Series]:
        """Return static held-out test features and labels."""
        return self._X_test.copy(), self._y_test.copy()

    def get_splits(self) -> Iterator[DataSplit]:
        """Yield DataSplit pairs across configured validation folds lazily."""
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
        elif v_cfg.method == ValidationMethod.K_FOLD:
            splitter = KFold(
                n_splits=v_cfg.n_splits,
                shuffle=v_cfg.shuffle,
                random_state=v_cfg.random_state,
            )
        elif v_cfg.method == ValidationMethod.STRATIFIED_K_FOLD:
            splitter = StratifiedKFold(
                n_splits=v_cfg.n_splits,
                shuffle=v_cfg.shuffle,
                random_state=v_cfg.random_state,
            )
        else:
            raise ValueError(f"Unsupported validation method: {v_cfg.method}")

        for fold, (train_idx, val_idx) in enumerate(
            splitter.split(self._X_dev, self._y_dev)
        ):
            yield DataSplit(
                X_train=self._X_dev.iloc[train_idx].reset_index(drop=True),
                y_train=self._y_dev.iloc[train_idx].reset_index(drop=True),
                X_val=self._X_dev.iloc[val_idx].reset_index(drop=True),
                y_val=self._y_dev.iloc[val_idx].reset_index(drop=True),
                fold=fold,
            )