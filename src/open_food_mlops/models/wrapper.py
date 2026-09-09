"""Custom MLflow PyFunc wrapper combining feature pre-processing and model inference."""

from __future__ import annotations

import logging
from typing import Any, Optional

import mlflow.pyfunc
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class NovaPipelineWrapper(mlflow.pyfunc.PythonModel):
    """Encapsulates feature transformer and estimator into a single MLflow PyFunc model."""

    def __init__(self, pipeline: Any, model: Any) -> None:
        """Initialize wrapper with fitted pipeline and model components."""
        self.pipeline = pipeline
        self.model = model

    def predict(
        self, context: mlflow.pyfunc.PythonModelContext, model_input: pd.DataFrame
    ) -> np.ndarray:
        """Transform input features and compute class predictions safely."""
        transformed_input = self.pipeline.transform(model_input)

        # Un-nest underlying scikit-learn estimator if using custom model wrapper
        estimator = getattr(self.model, "estimator_", self.model)
        estimator = getattr(self.model, "model", estimator)

        if hasattr(estimator, "predict_proba"):
            probs = estimator.predict_proba(transformed_input)
            classes = getattr(estimator, "classes_", None)
            best_idx = np.argmax(probs, axis=1)

            if classes is not None:
                return np.array([classes[i] for i in best_idx])
            return best_idx

        return estimator.predict(transformed_input)