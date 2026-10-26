"""Model-agnostic inference helpers shared by the REST API and batch pipelines.

Keeping prediction extraction, NOVA label mapping and probability validation in
one place guarantees that online, batch and reference-data predictions can never
drift apart.
"""

from __future__ import annotations

import logging
from typing import Any, Final

import numpy as np
import pandas as pd
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

MIN_RAW_CLASS: Final[int] = 0
MAX_RAW_CLASS: Final[int] = 4
_PROBABILITY_TOLERANCE: Final[float] = 1e-6

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]


def _candidate_estimators(model: Any) -> list[Any]:
    """Return objects to probe for ``predict_proba``, most specific first.

    MLflow PyFunc wrappers hide the underlying estimator, so we try to unwrap
    it before falling back to the wrapper itself.
    """
    candidates: list[Any] = []

    if hasattr(model, "unwrap_python_model"):
        try:
            unwrapped = model.unwrap_python_model()
        except Exception:  # noqa: BLE001 - unwrapping is best effort
            unwrapped = None
        if unwrapped is not None:
            candidates.append(unwrapped)

    model_impl = getattr(model, "_model_impl", None)
    if model_impl is not None:
        sklearn_model = getattr(model_impl, "sklearn_model", None)
        if sklearn_model is not None:
            candidates.append(sklearn_model)
        candidates.append(model_impl)

    candidates.append(model)
    return candidates


def predict_with_confidence(
    model: Any,
    features: pd.DataFrame,
) -> tuple[IntArray, FloatArray]:
    """Predict raw class labels and the probability of the predicted class.

    Args:
        model: Loaded MLflow PyFunc model (or any scikit-learn style estimator).
        features: Feature matrix containing one row per product.

    Returns:
        Tuple ``(raw_labels, confidences)``, both of length ``len(features)``.
        When the model exposes no probabilities, confidence is reported as 1.0.
    """
    for estimator in _candidate_estimators(model):
        if not hasattr(estimator, "predict_proba"):
            continue

        classes = getattr(estimator, "classes_", None)
        try:
            probabilities = estimator.predict_proba(features)
        except Exception as exc:  # noqa: BLE001 - try the next candidate
            logger.debug("predict_proba failed on %r: %s", estimator, exc)
            continue

        if isinstance(probabilities, pd.DataFrame):
            probabilities = probabilities.to_numpy()
        probabilities = np.asarray(probabilities, dtype=np.float64)

        best_indices = np.argmax(probabilities, axis=1)
        confidences = np.max(probabilities, axis=1)

        if classes is not None:
            labels = np.asarray(classes)[best_indices]
        else:
            labels = best_indices

        return np.asarray(labels).astype(np.int64), confidences

    predictions = model.predict(features)
    if isinstance(predictions, (pd.DataFrame, pd.Series)):
        predictions = predictions.to_numpy()
    labels = np.asarray(predictions).reshape(-1).astype(np.int64)
    return labels, np.ones(labels.shape, dtype=np.float64)


def to_nova_groups(raw_labels: IntArray | list[int]) -> IntArray:
    """Convert model class labels to the public NOVA 1-4 contract.

    Models are trained on zero-based classes (0-3), which map to NOVA 1-4.
    A label of 4 is treated as already being a public NOVA group.

    Raises:
        ValueError: If any label is outside the supported range.
    """
    labels = np.asarray(raw_labels, dtype=np.int64)

    if labels.size and (labels.min() < MIN_RAW_CLASS or labels.max() > MAX_RAW_CLASS):
        invalid = sorted({int(v) for v in labels if not MIN_RAW_CLASS <= v <= MAX_RAW_CLASS})
        raise ValueError(
            f"Model produced invalid NOVA classes {invalid}. "
            f"Expected raw classes {MIN_RAW_CLASS}-{MAX_RAW_CLASS - 1} "
            f"or public classes 1-{MAX_RAW_CLASS}."
        )

    return np.where(labels < MAX_RAW_CLASS, labels + 1, labels).astype(np.int64)


def validate_probabilities(confidences: FloatArray | list[float]) -> FloatArray:
    """Validate model confidences before they reach the API or the store.

    Values that exceed [0, 1] only through floating point noise are clipped.

    Raises:
        ValueError: If any value is non-finite or genuinely out of range.
    """
    values = np.asarray(confidences, dtype=np.float64)

    if not np.isfinite(values).all():
        raise ValueError("Model produced a non-finite probability.")

    if (values < -_PROBABILITY_TOLERANCE).any() or (values > 1.0 + _PROBABILITY_TOLERANCE).any():
        raise ValueError("Model produced a probability outside the range [0, 1].")

    return np.clip(values, 0.0, 1.0)
