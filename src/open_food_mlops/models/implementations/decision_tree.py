"""Decision Tree model implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Self

import joblib
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from ..base import BaseModel
from ..registry import register
from ..specs import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
    SearchSpace,
)

logger = logging.getLogger(__name__)


@register("decision_tree")
class DecisionTreeModel(BaseModel):
    """Decision Tree classifier adapter."""

    model_name = "decision_tree"

    @classmethod
    def get_default_params(cls) -> Mapping[str, Any]:
        """Return default Decision Tree parameters."""
        return {
            "criterion": "gini",
            "max_depth": None,
            "min_samples_split": 2,
            "min_samples_leaf": 1,
            "random_state": 42,
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Fit the Decision Tree classifier.

        Args:
            X: Training feature matrix.
            y: Training targets.

        Returns:
            The fitted model instance.
        """
        params = {**self.get_default_params(), **self.config}
        self.estimator_ = DecisionTreeClassifier(**params)
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
        """Return hyperparameter search space for Decision Tree."""
        return {
            "max_depth": IntParameter(low=2, high=30),
            "min_samples_split": IntParameter(low=2, high=20),
            "min_samples_leaf": IntParameter(low=1, high=20),
            "criterion": CategoricalParameter(choices=("gini", "entropy", "log_loss")),
            "ccp_alpha": FloatParameter(low=0.0, high=0.05),
        }

    def _save(self, path: Path) -> None:
        """Serialize the fitted Decision Tree estimator."""
        joblib.dump(
            {
                "config": self.config,
                "estimator": self.estimator_,
            },
            path / "model.joblib",
        )

    @classmethod
    def _load(cls, path: Path) -> Self:
        """Restore serialized Decision Tree model."""
        payload = joblib.load(path / "model.joblib")
        model = cls(payload["config"])
        model.estimator_ = payload["estimator"]
        model.is_fitted_ = True
        return model