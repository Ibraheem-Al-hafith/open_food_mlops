"""Abstract interfaces for machine-learning models.

The classes in this module define the contracts that every model
implementation in the project must satisfy.

Concrete models must not leak their implementation details into the
training pipeline. The pipeline interacts only with these abstractions.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, ClassVar, Self

import pandas as pd

logger = logging.getLogger(__name__)

from .specs import SearchSpace


class BaseModel(ABC):
    """Abstract interface for all project models.

    A concrete model is responsible for:

    - constructing its underlying estimator;
    - fitting the estimator;
    - generating predictions;
    - defining its hyperparameter search space;
    - serializing and restoring model state.

    The training pipeline should depend exclusively on this interface.

    Attributes:
        config: Model configuration.
        estimator_: Concrete underlying estimator after fitting.
        is_fitted_: Whether the model has been fitted.
    """

    model_name: ClassVar[str]

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
    ) -> None:
        """Initialize a model.

        Args:
            config: Model-specific configuration and parameters.
        """
        self.config: dict[str, Any] = dict(config or {})
        self.estimator_: Any | None = None
        self.is_fitted_: bool = False
        logger.debug("Initialized %s model with config: %s", self.__class__.__name__, self.config)

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Self:
        """Fit the model.

        Args:
            X: Training features.
            y: Training targets.

        Returns:
            The fitted model.
        """

    @abstractmethod
    def predict(
        self,
        X: pd.DataFrame,
    ) -> Any:
        """Generate predictions.

        Args:
            X: Input features.

        Returns:
            Model predictions.
        """

    @classmethod
    @abstractmethod
    def get_search_space(cls) -> SearchSpace:
        """Return the model's hyperparameter search space.

        The tuning engine consumes this specification without knowing
        anything about the underlying model.

        Returns:
            Model-specific hyperparameter search space.
        """

    @classmethod
    def get_default_params(cls) -> Mapping[str, Any]:
        """Return default model parameters.

        Concrete models can override this method.

        Returns:
            Default estimator parameters.
        """
        return {}

    def get_params(self) -> dict[str, Any]:
        """Return the current model configuration.

        Returns:
            Copy of the model configuration.
        """
        return dict(self.config)

    def set_params(
        self,
        **params: Any,
    ) -> Self:
        """Update model parameters.

        Args:
            **params: Parameters to update.

        Returns:
            The current model instance.
        """
        self.config.update(params)
        return self

    def save(
        self,
        path: str | Path,
    ) -> None:
        """Persist the fitted model.

        Concrete models implement the actual serialization mechanism.

        Args:
            path: Destination path.

        Raises:
            RuntimeError: If the model has not been fitted.
        """
        self._check_is_fitted()

        destination = Path(path)
        destination.mkdir(parents=True, exist_ok=True)
        logger.info("Saving model %s to %s", self.model_name, destination)

        self._save(destination)

    @classmethod
    def load(
        cls,
        path: str | Path,
    ) -> Self:
        """Restore a model from disk.

        Args:
            path: Serialized model path.

        Returns:
            Restored model instance.
        """
        logger.info("Loading model %s from %s", cls.__name__, path)
        return cls._load(Path(path))

    @abstractmethod
    def _save(
        self,
        path: Path,
    ) -> None:
        """Serialize implementation-specific model state."""

    @classmethod
    @abstractmethod
    def _load(
        cls,
        path: Path,
    ) -> Self:
        """Restore implementation-specific model state."""

    def _check_is_fitted(self) -> None:
        """Raise an error if the model has not been fitted."""
        if not self.is_fitted_:
            logger.error("Attempted to use unfitted model %s.", self.model_name)
            raise RuntimeError(
                f"Model {self.model_name!r} has not been fitted. "
                "Call fit() before this operation."
            )