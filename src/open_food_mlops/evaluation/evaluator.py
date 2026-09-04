"""Evaluation metrics calculation engine."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class EvaluationResult:
    """Contains calculated metrics and execution metadata."""

    metrics: dict[str, float]
    primary_metric_name: str = "macro_f1"

    @property
    def primary_score(self) -> float:
        """Extract primary metric score."""
        return self.metrics.get(self.primary_metric_name, 0.0)


class Evaluator:
    """Computes multi-class classification evaluation metrics."""

    def __init__(self, primary_metric: str = "macro_f1") -> None:
        self.primary_metric = primary_metric

    def evaluate(self, y_true: pd.Series, y_pred: pd.Series) -> EvaluationResult:
        """Compute evaluation metrics comparing ground truth with predictions."""
        logger.debug("Evaluating predictions over %d samples.", len(y_true))
        acc = float(accuracy_score(y_true, y_pred))
        prec = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
        rec = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
        f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))

        metrics = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "macro_f1": f1,
        }

        logger.info(
            "Evaluation metrics computed: accuracy=%.4f, precision=%.4f, recall=%.4f, macro_f1=%.4f",
            acc,
            prec,
            rec,
            f1,
        )
        return EvaluationResult(
            metrics=metrics, primary_metric_name=self.primary_metric
        )