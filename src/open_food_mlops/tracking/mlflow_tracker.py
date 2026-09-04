"""MLflow tracking implementation configured with explicit serialization format."""

from __future__ import annotations

import logging
from typing import Any

import mlflow
import mlflow.sklearn

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Handles interaction with MLflow tracking and model registry services."""

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        """Initialize MLflow tracking URI and experiment context."""
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    def start_run(self, run_name: str) -> mlflow.ActiveRun:
        """Start a new active MLflow run context."""
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        """Log key-value hyperparameter dictionary."""
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        """Log evaluation metric values."""
        mlflow.log_metrics(metrics)

    def log_artifact(self, local_path: str) -> None:
        """Log local directory or artifact path."""
        mlflow.log_artifact(local_path)

    def register_model(
        self,
        model: Any,
        artifact_path: str,
        registered_model_name: str,
    ) -> None:
        """Explicitly log and register a model instance into MLflow Model Registry.

        Uses cloudpickle serialization to safely handle custom wrapped classes and Joblib pipelines.
        """
        logger.info(
            "Registering model artifact under registry name: '%s'",
            registered_model_name,
        )

        try:
            # Explicitly set serialization_format to cloudpickle to bypass skops restriction
            mlflow.sklearn.log_model(
                sk_model=model,
                name=artifact_path,
                registered_model_name=registered_model_name,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )
        except TypeError:
            # Fallback for MLflow versions using artifact_path instead of name kwarg
            mlflow.sklearn.log_model(
                sk_model=model,
                artifact_path=artifact_path,
                registered_model_name=registered_model_name,
                serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
            )