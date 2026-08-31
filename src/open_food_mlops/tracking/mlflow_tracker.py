"""MLflow experiment tracking adapter."""

from __future__ import annotations

from typing import Any
import mlflow


class MLflowTracker:
    """Wrapper around MLflow client for experiment logging."""

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str) -> mlflow.ActiveRun:
        """Context manager to start an active MLflow run."""
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log key-value hyperparameter dictionary."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float], step: int | None = None) -> None:
        """Log evaluation metrics."""
        mlflow.log_metrics(metrics, step=step)

    def log_artifact(self, local_path: str) -> None:
        """Persist directory or file artifact to MLflow."""
        mlflow.log_artifact(local_path)