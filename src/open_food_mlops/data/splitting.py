"""Dataset splitting utilities for machine learning pipelines."""

from __future__ import annotations

import logging
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

logger = logging.getLogger(__name__)


class ValidationMethod(StrEnum):
    """Supported validation strategies."""

    TRAIN_TEST_SPLIT = "train_test_split"
    K_FOLD = "kfold"
    STRATIFIED_K_FOLD = "stratified_kfold"


@dataclass(frozen=True, slots=True)
class TestConfig:
    """Configuration for hold-out test set partition."""

    test_size: float = 0.2
    stratify: bool = True
    random_state: int = 42


@dataclass(frozen=True, slots=True)
class ValidationConfig:
    """Configuration for validation splitting strategy."""

    method: ValidationMethod = ValidationMethod.STRATIFIED_K_FOLD
    n_splits: int = 5
    validation_size: float = 0.2
    shuffle: bool = True
    stratify: bool = True
    random_state: int = 42


@dataclass(frozen=True, slots=True)
class DataSplitConfig:
    """Master configuration for dataset sampling and partitioning."""

    sample_fraction: float = 1.0
    test: TestConfig = field(default_factory=TestConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)


@dataclass(frozen=True, slots=True)
class DatasetSplit:
    """Container representing a single train and validation partition pair."""

    X_train: pd.DataFrame
    y_train: pd.Series
    X_validation: pd.DataFrame
    y_validation: pd.Series


@dataclass
class DatasetSplits:
    """Holds development and final test partitions of a dataset."""

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
        """Construct dataset partitions with optional stratified fraction sampling."""
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError("dataframe must be a pandas DataFrame.")
        if dataframe.empty:
            raise ValueError("Cannot split an empty DataFrame.")
        if target not in dataframe.columns:
            raise ValueError(f"Target column '{target}' non-existent in DataFrame.")

        config = config or DataSplitConfig()

        # Perform stratified subsampling if fraction < 1.0
        if 0.0 < config.sample_fraction < 1.0:
            logger.info("Sampling %s fraction of dataset (stratified)", config.sample_fraction)
            _, sample_df = train_test_split(
                dataframe,
                test_size=config.sample_fraction,
                random_state=config.test.random_state,
                stratify=dataframe[target],
            )
            dataframe = sample_df.reset_index(drop=True)

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
        v_cfg = self.config.validation

        if v_cfg.method == ValidationMethod.TRAIN_TEST_SPLIT:
            splitter_cls = StratifiedShuffleSplit if v_cfg.stratify else ShuffleSplit
            splitter = splitter_cls(
                n_splits=1,
                test_size=v_cfg.validation_size,
                random_state=v_cfg.random_state,
            )
            return splitter.split(self.X_train, self.y_train)

        if v_cfg.method == ValidationMethod.K_FOLD:
            return KFold(
                n_splits=v_cfg.n_splits,
                shuffle=v_cfg.shuffle,
                random_state=v_cfg.random_state,
            ).split(self.X_train, self.y_train)

        if v_cfg.method == ValidationMethod.STRATIFIED_K_FOLD:
            return StratifiedKFold(
                n_splits=v_cfg.n_splits,
                shuffle=v_cfg.shuffle,
                random_state=v_cfg.random_state,
            ).split(self.X_train, self.y_train)

        raise ValueError(f"Unsupported validation method: {v_cfg.method!r}")

    @property
    def splits(self) -> Iterator[DatasetSplit]:
        """Lazily yield DatasetSplit pairs for cross-validation."""
        for train_idx, val_idx in self._get_splitter():
            yield DatasetSplit(
                X_train=self.X_train.iloc[train_idx].reset_index(drop=True),
                y_train=self.y_train.iloc[train_idx].reset_index(drop=True),
                X_validation=self.X_train.iloc[val_idx].reset_index(drop=True),
                y_validation=self.y_train.iloc[val_idx].reset_index(drop=True),
            )