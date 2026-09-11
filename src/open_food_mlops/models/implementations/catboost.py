"""CatBoost model implementation compatible with CatBoost >= 1.2.10."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Self

import joblib
import pandas as pd

try:
    import catboost as cb
except ImportError:
    cb = None  # type: ignore[assignment]

from ..base import BaseModel
from ..registry import register
from ..specs import (
    FloatParameter,
    IntParameter,
    SearchSpace,
)

logger = logging.getLogger(__name__)


@register("catboost")
class CatBoostModel(BaseModel):
    """CatBoost Classifier adapter satisfying the BaseModel interface."""

    model_name = "catboost"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        if cb is None:
            raise ImportError(
                "CatBoost is not installed. Please install it via `pip install catboost`."
            )
        super().__init__(config)

    @classmethod
    def get_default_params(cls) -> Mapping[str, Any]:
        """Return default CatBoost parameters."""
        return {
            "iterations": 200,
            "learning_rate": 0.08,
            "depth": 6,
            "random_seed": 42,
            "verbose": 0,
            "loss_function": "MultiClass",
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Fit the CatBoost classifier."""
        params = {**self.get_default_params(), **self.config}
        assert cb is not None
        
        num_classes = len(y.unique())
        if num_classes == 2:
            params["loss_function"] = "Logloss"

        self.estimator_ = cb.CatBoostClassifier(**params)
        self.estimator_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate class predictions."""
        self._check_is_fitted()
        assert self.estimator_ is not None
        predictions = self.estimator_.predict(X)
        
        # Flatten predictions array if multi-dimensional
        if hasattr(predictions, "ndim") and predictions.ndim > 1:
            predictions = predictions.ravel()
            
        return pd.Series(
            predictions,
            index=X.index,
            name="prediction",
        )

    def predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        """Generate class probabilities."""
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
        """Return hyperparameter search space for CatBoost."""
        return {
            "iterations": IntParameter(low=100, high=1000, step=50),
            "depth": IntParameter(low=4, high=10),
            "learning_rate": FloatParameter(low=0.01, high=0.3, log=True),
            "l2_leaf_reg": FloatParameter(low=1.0, high=10.0),
        }

    def _save(self, path: Path) -> None:
        """Serialize the fitted CatBoost estimator."""
        joblib.dump(
            {
                "config": self.config,
                "estimator": self.estimator_,
            },
            path / "model.joblib",
        )

    @classmethod
    def _load(cls, path: Path) -> Self:
        """Restore serialized CatBoost model."""
        payload = joblib.load(path / "model.joblib")
        model = cls(payload["config"])
        model.estimator_ = payload["estimator"]
        model.is_fitted_ = True
        return model