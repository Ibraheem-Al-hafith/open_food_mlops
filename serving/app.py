"""Production FastAPI serving application for inference execution."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator, Dict, Tuple

import mlflow
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response, status

from open_food_mlops.config.features import FEATURE_COLUMNS
from open_food_mlops.config.settings import settings
from serving.metrics import PREDICTION_COUNTER, PREDICTION_LATENCY, setup_monitoring
from serving.prediction_store import prediction_store
from serving.schemas import NovaPredictRequest, NovaPredictResponse

logger = logging.getLogger(__name__)

MODEL_CONTAINER: Dict[str, Any] = {}


def _load_champion_model(
    tracking_uri: str,
    registered_model_name: str = "open_food_champion",
    alias: str = "production",
) -> Tuple[Any, str]:
    """Load model from MLflow Model Registry via explicit governance aliases."""
    mlflow.set_tracking_uri(tracking_uri)
    model_uri = f"models:/{registered_model_name}@{alias}"

    logger.info("Loading champion model from registry alias URI: %s", model_uri)

    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
    except Exception as exc:
        logger.error(
            "Failed to load production model %s: %s",
            model_uri,
            exc,
            exc_info=True,
        )
        raise RuntimeError(
            f"Production model '{model_uri}' could not be loaded."
        ) from exc

    return loaded_model, alias


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage application startup lifecycle and champion loading."""
    logger.info("Initializing REST Serving Layer...")
    try:
        model, version = _load_champion_model(
            tracking_uri=settings.mlflow_tracking_uri,
        )
        MODEL_CONTAINER["champion"] = model
        MODEL_CONTAINER["model_version"] = version
        logger.info("Champion model [%s] loaded successfully.", version)
    except Exception as exc:
        logger.error("Champion model initialization failed: %s", exc)
        MODEL_CONTAINER["champion"] = None
        MODEL_CONTAINER["model_version"] = "unknown"

    yield
    MODEL_CONTAINER.clear()


app = FastAPI(
    title="Open Food MLOps Serving API",
    version="1.0.0",
    lifespan=lifespan,
)

setup_monitoring(app)


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check(response: Response) -> Dict[str, Any]:
    """Service readiness health check returning 503 if model unready."""
    is_ready = MODEL_CONTAINER.get("champion") is not None
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return {"status": "degraded", "model_loaded": False}
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": MODEL_CONTAINER.get("model_version", "unknown"),
    }


def _extract_probabilities_and_pred(
    champion: Any, input_data: pd.DataFrame
) -> Tuple[int, float]:
    """Safely extract predicted index and probability using estimator classes_."""
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
        classes = getattr(estimator, "classes_", None)
        if hasattr(estimator, "predict_proba"):
            try:
                probs = estimator.predict_proba(input_data)
                if isinstance(probs, pd.DataFrame):
                    probs = probs.to_numpy()
                best_idx = int(np.argmax(probs[0]))
                pred_label = int(classes[best_idx]) if classes is not None else best_idx
                return pred_label, float(probs[0][best_idx])
            except Exception as exc:
                logger.debug("predict_proba failed on candidate %s: %s", estimator, exc)
                continue

    preds = champion.predict(input_data)
    if isinstance(preds, pd.DataFrame):
        preds = preds.to_numpy()

    raw_pred = int(preds[0]) if isinstance(preds, (np.ndarray, list)) else int(preds)
    return raw_pred, 1.0



def _normalize_nova_group(raw_pred: int) -> int:
    """Convert model class labels to the public NOVA 1-4 contract."""
    if raw_pred in (0, 1, 2, 3):
        return raw_pred + 1

    if raw_pred in (1, 2, 3, 4):
        return raw_pred

    raise ValueError(
        f"Model produced invalid NOVA class '{raw_pred}'. "
        "Expected raw classes 0-3 or public classes 1-4."
    )


def _validate_confidence(confidence: float) -> float:
    """Validate model confidence before it reaches the API/storage layer."""
    confidence = float(confidence)

    if not np.isfinite(confidence):
        raise ValueError("Model produced a non-finite probability.")

    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            f"Model produced invalid probability '{confidence}'. "
            "Expected a value between 0 and 1."
        )

    return confidence

@app.post(
    "/v1/predict",
    response_model=NovaPredictResponse,
    status_code=status.HTTP_200_OK,
)
def predict(
    request: NovaPredictRequest,
    background_tasks: BackgroundTasks,
) -> NovaPredictResponse:
    """Execute real-time NOVA classification following strict schema enforcement."""
    champion = MODEL_CONTAINER.get("champion")
    if champion is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable.",
        )

    try:
        start_time = time.perf_counter()
        raw_dict = request.model_dump(by_alias=True)
        product_code = raw_dict.get("product_code", None)

        input_df = pd.DataFrame([raw_dict])[FEATURE_COLUMNS].astype(float)
                
        raw_pred, confidence = _extract_probabilities_and_pred(
            champion,
            input_df,
        )

        nova_group = _normalize_nova_group(raw_pred)
        confidence = _validate_confidence(confidence)

        PREDICTION_LATENCY.observe(time.perf_counter() - start_time)
        PREDICTION_COUNTER.labels(nova_group=str(nova_group)).inc()

        background_tasks.add_task(
            prediction_store.record_prediction,
            features=input_df.iloc[0].to_dict(),
            prediction=nova_group,
            probability=round(confidence, 4),
            model_version=MODEL_CONTAINER.get("model_version"),
            product_code=product_code,
        )

        return NovaPredictResponse(
            nova_group=nova_group,
            probability=round(confidence, 4),
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error(
            "Inference execution error: %s",
            exc,
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Inference failed.",
        ) from exc