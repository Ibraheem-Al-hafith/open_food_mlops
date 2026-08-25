"""XGBoost model implementation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Self

import joblib
import pandas as pd

try:
    import xgboost as xgb
except ImportError:
    xgb = None  # type: ignore[assignment]

from ..base import BaseModel
from ..registry import register
from ..specs import (
    CategoricalParameter,
    FloatParameter,
    IntParameter,
    SearchSpace,
)

logger = logging.getLogger(__name__)


@register("xgboost")
class XGBoostModel(BaseModel):
    """XGBoost Classifier adapter satisfying the BaseModel interface.

    Handles multi-class and binary classification using XGBoost.
    """

    model_name = "xgboost"

    def __init__(self, config: Mapping[str, Any] | None = None) -> None:
        """Initialize the XGBoost model wrapper.

        Raises:
            ImportError: If the xgboost library is not installed.
        """
        if xgb is None:
            raise ImportError(
                "XGBoost is not installed. Please install it via `pip install xgboost`."
            )
        super().__init__(config)

    @classmethod
    def get_default_params(cls) -> Mapping[str, Any]:
        """Return default XGBoost parameters."""
        return {
            "n_estimators": 100,
            "learning_rate": 0.1,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": 42,
            "n_jobs": -1,
            "eval_metric": "logloss",
        }

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        """Fit the XGBoost classifier.

        Args:
            X: Training feature matrix.
            y: Training targets.

        Returns:
            The fitted model instance.
        """
        params = {**self.get_default_params(), **self.config}
        assert xgb is not None
        self.estimator_ = xgb.XGBClassifier(**params)
        self.estimator_.fit(X, y)
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> pd.Series:
        """Generate class predictions.

        Args:
            X: Input feature matrix.

        Returns:
            pd.Series containing class predictions indexed by X.index.
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
            pd.DataFrame containing prediction probabilities for each class.
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
        """Return hyperparameter search space for tuning XGBoost."""
        return {
            "n_estimators": IntParameter(low=50, high=1000, step=50),
            "max_depth": IntParameter(low=3, high=12),
            "learning_rate": FloatParameter(low=0.005, high=0.3, log=True),
            "subsample": FloatParameter(low=0.5, high=1.0),
            "colsample_bytree": FloatParameter(low=0.5, high=1.0),
            "min_child_weight": IntParameter(low=1, high=10),
            "gamma": FloatParameter(low=0.0, high=5.0),
            "booster": CategoricalParameter(choices=("gbtree", "dart")),
        }

    def _save(self, path: Path) -> None:
        """Serialize the fitted XGBoost estimator and configuration."""
        metadata = {
            "config": self.config,
            "classes": getattr(self.estimator_, "classes_", None),
        }
        joblib.dump(metadata, path / "metadata.joblib")
        assert self.estimator_ is not None
        self.estimator_.save_model(path / "model.json")

    @classmethod
    def _load(cls, path: Path) -> Self:
        """Restore serialized XGBoost model state."""
        metadata = joblib.load(path / "metadata.joblib")
        model = cls(metadata["config"])
        assert xgb is not None

        estimator = xgb.XGBClassifier()
        estimator.load_model(path / "model.json")
        if metadata.get("classes") is not None:
            estimator.classes_ = metadata["classes"]

        model.estimator_ = estimator
        model.is_fitted_ = True
        return model