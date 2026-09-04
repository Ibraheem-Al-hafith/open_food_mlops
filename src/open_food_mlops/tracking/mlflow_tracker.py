"""MLflow experiment tracking adapter."""

from __future__ import annotations

import logging
from typing import Any

import mlflow

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Wrapper around MLflow client for experiment logging."""

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        logger.info("Configuring MLflow tracker with URI=%s and experiment=%s", tracking_uri, experiment_name)
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str) -> mlflow.ActiveRun:
        """Context manager to start an active MLflow run."""
        logger.info("Starting MLflow run: %s", run_name)
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log key-value hyperparameter dictionary."""
        logger.info("Logging params to MLflow: %s", params)
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log evaluation metrics."""
        logger.info("Logging metrics to MLflow: %s (step=%s)", metrics, step)
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str) -> None:
        """Persist directory or file artifact to MLflow."""
        logger.info("Logging artifact to MLflow from %s", local_path)
        mlflow.log_artifact(local_path)