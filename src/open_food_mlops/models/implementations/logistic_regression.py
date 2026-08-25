"""Logistic Regression model implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Self

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression

from ..base import BaseModel
from ..registry import register
from ..specs import (
    CategoricalParameter,
    FloatParameter,
    SearchSpace,
)

logger = logging.getLogger(__name__)


@register("logistic_regression")
class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier adapter."""

    model_name = "logistic_regression"

    @classmethod
    def get_default_params(cls) -> Mapping[str, Any]:
        """Return default Logistic Regression parameters."""
        return {
            "C": 1.0,
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
            "random_state": 42,
            "n_jobs": -1,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Fit the Logistic Regression classifier.

        Args:
            X: Training feature matrix.
            y: Training targets.

        Returns:
            The fitted model instance.
        """
        params = {**self.get_default_params(), **self.config}
        self.estimator_ = LogisticRegression(**params)
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
        """Return hyperparameter search space for Logistic Regression."""
        return {
            "C": FloatParameter(low=1e-4, high=100.0, log=True),
            "penalty": CategoricalParameter(choices=("l1", "l2")),
            "solver": CategoricalParameter(choices=("saga", "liblinear", "lbfgs")),
        }

    def _save(self, path: Path) -> None:
        """Serialize the fitted Logistic Regression estimator."""
        joblib.dump(
            {
                "config": self.config,
                "estimator": self.estimator_,
            },
            path / "model.joblib",
        )

    @classmethod
    def _load(cls, path: Path) -> Self:
        """Restore serialized Logistic Regression model."""
        payload = joblib.load(path / "model.joblib")
        model = cls(payload["config"])
        model.estimator_ = payload["estimator"]
        model.is_fitted_ = True
        return model