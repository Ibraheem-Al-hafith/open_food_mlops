"""Production FastAPI serving application for inference execution."""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
from typing import Any, AsyncGenerator, Dict, Tuple

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from open_food_mlops.config.settings import settings
from serving.schemas import NovaPredictRequest, NovaPredictResponse

logger = logging.getLogger(__name__)

# Server-wide model context holder
MODEL_CONTAINER: Dict[str, Any] = {}


def _load_champion_model(
    tracking_uri: str,
    experiment_name: str,
) -> Any:
    """Dynamically discover and load the top-performing finished model from MLflow.

    Args:
        tracking_uri: Local or remote MLflow tracking URI.
        experiment_name: MLflow experiment namespace.

    Returns:
        Loaded pyfunc MLflow model.

    Raises:
        RuntimeError: If experiment or usable model artifacts cannot be resolved.
    """
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()

    experiment = client.get_experiment_by_name(experiment_name)
    if not experiment:
        raise RuntimeError(f"MLflow experiment '{experiment_name}' does not exist.")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.macro_f1 DESC"],
        max_results=5,
    )

    if not runs:
        raise RuntimeError(
            f"No finished runs found in MLflow experiment '{experiment_name}'."
        )

    for run in runs:
        run_id = run.info.run_id
        artifacts = client.list_artifacts(run_id)

        model_subpath = None
        for art in artifacts:
            if art.is_dir:
                dir_contents = [
                    f.path for f in client.list_artifacts(run_id, art.path)
                ]
                if any("MLmodel" in path for path in dir_contents):
                    model_subpath = art.path
                    break

        if not model_subpath:
            if any("MLmodel" in art.path for art in artifacts):
                model_uri = f"runs:/{run_id}"
            else:
                model_uri = f"runs:/{run_id}/model"
        else:
            model_uri = f"runs:/{run_id}/{model_subpath}"

        try:
            logger.info("Attempting to load model from MLflow URI: %s", model_uri)
            return mlflow.pyfunc.load_model(model_uri)
        except Exception as exc:
            logger.warning("Failed loading model from URI %s: %s", model_uri, exc)
            continue

    raise RuntimeError(
        f"Unable to load a valid MLflow model artifact from top runs in '{experiment_name}'."
    )


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup and model loading lifecycle."""
    logger.info("Initializing REST Serving Layer...")
    try:
        MODEL_CONTAINER["champion"] = _load_champion_model(
            tracking_uri=settings.mlflow_tracking_uri,
            experiment_name=settings.mlflow_experiment_name,
        )
        logger.info("Champion model successfully loaded and ready for inference.")
    except Exception as exc:
        logger.error("Champion model initialization failed: %s", exc)
        MODEL_CONTAINER["champion"] = None

    yield
    MODEL_CONTAINER.clear()


app = FastAPI(
    title="Open Food MLOps Serving API",
    version="0.2.2",
    lifespan=lifespan,
)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, str]:
    """Health check status endpoint."""
    is_ready = MODEL_CONTAINER.get("champion") is not None
    return {
        "status": "healthy" if is_ready else "degraded",
        "model_loaded": str(is_ready),
    }


def _extract_probabilities_and_pred(
    champion: Any, input_data: pd.DataFrame
) -> Tuple[int, float]:
    """Extract prediction and probability safely from MLflow PyFunc or underlying flavor."""
    candidate_estimators = []

    if hasattr(champion, "unwrap_python_model"):
        try:
            unwrapped = champion.unwrap_python_model()
            if unwrapped is not None:
                candidate_estimators.append(unwrapped)
        except Exception:
            pass

    model_impl = getattr(champion, "_model_impl", None)
    if model_impl is not None:
        sklearn_model = getattr(model_impl, "sklearn_model", None)
        if sklearn_model is not None:
            candidate_estimators.append(sklearn_model)
        candidate_estimators.append(model_impl)

    candidate_estimators.append(champion)

    for estimator in candidate_estimators:
        if hasattr(estimator, "predict_proba"):
            try:
                probs = estimator.predict_proba(input_data)
                if isinstance(probs, pd.DataFrame):
                    probs = probs.to_numpy()
                best_idx = int(np.argmax(probs[0]))
                return best_idx, float(probs[0][best_idx])
            except Exception as exc:
                logger.debug("Failed predict_proba call on candidate %s: %s", estimator, exc)
                continue

    preds = champion.predict(input_data)
    if isinstance(preds, pd.DataFrame):
        preds = preds.to_numpy()

    if isinstance(preds, np.ndarray) and preds.ndim == 2 and preds.shape[1] > 1:
        best_idx = int(np.argmax(preds[0]))
        return best_idx, float(preds[0][best_idx])

    raw_pred = int(preds[0]) if isinstance(preds, (np.ndarray, list)) else int(preds)
    return raw_pred, 1.0


@app.post(
    "/predict",
    response_model=NovaPredictResponse,
    status_code=status.HTTP_200_OK,
)
def predict(request: NovaPredictRequest) -> NovaPredictResponse:
    """Execute real-time NOVA group classification using loaded champion model."""
    champion = MODEL_CONTAINER.get("champion")
    if champion is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Champion model is not available. Check server logs or MLflow database.",
        )

    try:
        input_data = pd.DataFrame([request.model_dump(by_alias=True)])
        raw_pred, confidence = _extract_probabilities_and_pred(champion, input_data)
        nova_group = raw_pred + 1 if raw_pred < 4 else raw_pred

        return NovaPredictResponse(
            nova_group=nova_group,
            probability=round(confidence, 4),
        )

    except Exception as exc:
        logger.error("Inference execution error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference processing error: {str(exc)}",
        ) from exc