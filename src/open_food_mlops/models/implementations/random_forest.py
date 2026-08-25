"""Random Forest model implementation."""

from __future__ import annotations

import joblib
from pathlib import Path
from typing import Self

import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from ..base import BaseModel
from ..registry import register
from ..specs import (
    CategoricalParameter,
    IntParameter,
    SearchSpace,
)


@register("random_forest")
class RandomForestModel(BaseModel):
    """Random Forest classifier adapter."""

    model_name = "random_forest"

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Self:
        """Fit the Random Forest classifier."""
        self.estimator_ = RandomForestClassifier(
            **self.config,
        )

        self.estimator_.fit(X, y)

        self.is_fitted_ = True

        return self

    def predict(
        self,
        X: pd.DataFrame,
    ) -> pd.Series:
        """Generate class predictions."""
        self._check_is_fitted()

        assert self.estimator_ is not None
        predictions = self.estimator_.predict(X)

        return pd.Series(
            predictions,
            index=X.index,
            name="prediction",
        )

    @classmethod
    def get_search_space(cls) -> SearchSpace:
        """Return Random Forest hyperparameter search space."""
        return {
            "n_estimators": IntParameter(
                low=100,
                high=1000,
            ),
            "max_depth": IntParameter(
                low=3,
                high=50,
            ),
            "criterion": CategoricalParameter(
                choices=(
                    "gini",
                    "entropy",
                    "log_loss",
                ),
            ),
        }

    def _save(
        self,
        path: Path,
    ) -> None:
        """Serialize the fitted estimator."""
        joblib.dump(
            {
                "config": self.config,
                "estimator": self.estimator_,
            },
            path / "model.joblib",
        )

    @classmethod
    def _load(
        cls,
        path: Path,
    ) -> Self:
        """Restore a serialized Random Forest."""
        payload = joblib.load(path / "model.joblib")

        model = cls(payload["config"])
        model.estimator_ = payload["estimator"]
        model.is_fitted_ = True

        return model