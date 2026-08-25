"""LightGBM model implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Self

import joblib
import pandas as pd

try:
    import lightgbm as lgb
except ImportError:
    lgb = None  # type: ignore[assignment]

from ..base import BaseModel
from ..registry import register
from ..specs import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
    SearchSpace,
)

logger = logging.getLogger(__name__)


@register("lightgbm")
class LightGBMModel(BaseModel):
    """LightGBM Classifier adapter satisfying the BaseModel interface."""

    model_name = "lightgbm"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        """Initialize the LightGBM model wrapper.

        Raises:
            ImportError: If the lightgbm library is not installed.
        """
        if lgb is None:
            raise ImportError(
                "LightGBM is not installed. Please install it via `pip install lightgbm`."
            )
        super().__init__(config)

    @classmethod
    def get_default_params(cls) -> Mapping[str, Any]:
        """Return default LightGBM parameters."""
        return {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "num_leaves": 31,
            "max_depth": -1,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Fit the LightGBM classifier.

        Args:
            X: Training feature matrix.
            y: Training targets.

        Returns:
            The fitted model instance.
        """
        params = {**self.get_default_params(), **self.config}
        assert lgb is not None
        self.estimator_ = lgb.LGBMClassifier(**params)
        self.estimator_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate class predictions.

        Args:
            X: Input feature matrix.

        Returns:
            pd.Series containing class predictions.
        """
        self._check_is_fitted()
        assert self.estimator_ is not None
        predictions = self.estimator_.predict(X)
        return pd.Series(
            predictions,
            index=X.index,
            name="prediction",
        )

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate class probabilities.

        Args:
            X: Input feature matrix.

        Returns:
            pd.DataFrame with columns for each class probability.
        """
        self._check_is_fitted()
        assert self.estimator_ is not None
        probabilities = self.estimator_.predict_proba(X)
        classes = self.estimator_.classes_
        return pd.DataFrame(
            probabilities,
            index=X.index,
            columns=[f"prob_{c}" for c in classes],
        )

    @classmethod
    def get_search_space(cls) -> SearchSpace:
        """Return hyperparameter search space for tuning LightGBM."""
        return {
            "n_estimators": IntParameter(low=50, high=1000, step=50),
            "num_leaves": IntParameter(low=15, high=255),
            "max_depth": IntParameter(low=3, high=15),
            "learning_rate": FloatParameter(low=0.005, high=0.3, log=True),
            "subsample": FloatParameter(low=0.5, high=1.0),
            "colsample_bytree": FloatParameter(low=0.5, high=1.0),
            "min_child_samples": IntParameter(low=5, high=100),
            "boosting_type": CategoricalParameter(choices=("gbdt", "dart")),
        }

    def _save(self, path: Path) -> None:
        """Serialize the fitted LightGBM estimator."""
        joblib.dump(
            {
                "config": self.config,
                "estimator": self.estimator_,
            },
            path / "model.joblib",
        )

    @classmethod
    def _load(cls, path: Path) -> Self:
        """Restore serialized LightGBM model."""
        payload = joblib.load(path / "model.joblib")
        model = cls(payload["config"])
        model.estimator_ = payload["estimator"]
        model.is_fitted_ = True
        return model