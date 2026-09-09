"""MLflow tracker implementation supporting PyFunc model logging and alias governance."""

from __future__ import annotations

import logging
from typing import Any, Optional

import mlflow

logger = logging.getLogger(__name__)


class MLflowTracker:
    """Encapsulates tracking operations and model registry operations."""

    def __init__(self, tracking_uri: str, experiment_name: str) -> None:
        self.tracking_uri = tracking_uri
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)
        self.client = mlflow.tracking.MlflowClient()

    def start_run(self, run_name: Optional[str] = None) -> Any:
        return mlflow.start_run(run_name=run_name)

    def log_params(self, params: dict[str, Any]) -> None:
        mlflow.log_params(params)

    def log_metrics(self, metrics: dict[str, float]) -> None:
        mlflow.log_metrics(metrics)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None) -> None:
        mlflow.log_artifact(local_path, artifact_path=artifact_path)

    def register_and_alias_pyfunc(
        self,
        pyfunc_model: mlflow.pyfunc.PythonModel,
        artifact_path: str,
        registered_model_name: str,
        alias: str = "production",
    ) -> str:
        """Log PyFunc model, register in MLflow Model Registry, and set production alias."""
        model_info = mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=pyfunc_model,
            registered_model_name=registered_model_name,
        )

        model_version = model_info.registered_model_version
        if model_version:
            logger.info(
                "Setting alias '%s' for registered model '%s' (version %s)",
                alias,
                registered_model_name,
                model_version,
            )
            self.client.set_registered_model_alias(
                name=registered_model_name,
                alias=alias,
                version=str(model_version),
            )
        return str(model_version)