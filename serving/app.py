"""Production FastAPI serving application for inference execution."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, Optional, Sequence, Tuple

import mlflow
import numpy as np
import pandas as pd
from fastapi import BackgroundTasks, FastAPI, HTTPException, Response, status
from pydantic import ValidationError

from open_food_mlops.config.features import FEATURE_COLUMNS
from open_food_mlops.config.settings import settings
from serving.inference import (
    predict_with_confidence,
    to_nova_groups,
    validate_probabilities,
)
from serving.metrics import PREDICTION_COUNTER, PREDICTION_LATENCY, setup_monitoring
from serving.prediction_store import prediction_store
from serving.schemas import (
    BatchPrediction,
    BatchRowError,
    NovaBatchPredictRequest,
    NovaBatchPredictResponse,
    NovaPredictRequest,
    NovaPredictResponse,
)

logger = logging.getLogger(__name__)

MODEL_CONTAINER: Dict[str, Any] = {}

_MAX_ERROR_ITEMS = 3


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
    version="1.1.0",
    lifespan=lifespan,
)

setup_monitoring(app)


def _require_champion() -> Any:
    """Return the loaded champion model or raise HTTP 503."""
    champion = MODEL_CONTAINER.get("champion")
    if champion is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is unavailable.",
        )
    return champion


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
    champion = _require_champion()

    try:
        start_time = time.perf_counter()
        raw_dict = request.model_dump(by_alias=True)
        product_code = raw_dict.get("product_code", None)

        input_df = pd.DataFrame([raw_dict])[FEATURE_COLUMNS].astype(float)

        raw_labels, confidences = predict_with_confidence(champion, input_df)
        nova_group = int(to_nova_groups(raw_labels)[0])
        confidence = float(validate_probabilities(confidences)[0])

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


@dataclass(frozen=True, slots=True)
class _ValidatedRecord:
    """A batch record that passed schema validation."""

    index: int
    payload: NovaPredictRequest


def _summarize_validation_error(exc: ValidationError) -> str:
    """Condense a pydantic error into a short, human readable message."""
    errors = exc.errors()
    parts = [
        f"{'.'.join(str(p) for p in err['loc']) or 'record'}: {err['msg']}"
        for err in errors[:_MAX_ERROR_ITEMS]
    ]
    remaining = len(errors) - _MAX_ERROR_ITEMS
    if remaining > 0:
        parts.append(f"... and {remaining} more")
    return "; ".join(parts)


def _validate_records(
    records: Sequence[Dict[str, Any]],
) -> Tuple[list[_ValidatedRecord], list[BatchRowError]]:
    """Validate records independently so one bad row never fails the batch."""
    valid: list[_ValidatedRecord] = []
    errors: list[BatchRowError] = []

    for index, record in enumerate(records):
        try:
            valid.append(
                _ValidatedRecord(index, NovaPredictRequest.model_validate(record))
            )
        except ValidationError as exc:
            errors.append(
                BatchRowError(index=index, detail=_summarize_validation_error(exc))
            )

    return valid, errors


def _persist_batch(
    features: pd.DataFrame,
    nova_groups: np.ndarray,
    probabilities: np.ndarray,
    product_codes: Sequence[Optional[str]],
    model_version: Optional[str],
) -> None:
    """Write batch predictions to the production store (runs as background task)."""
    failures = 0
    for position, (_, row) in enumerate(features.iterrows()):
        try:
            prediction_store.record_prediction(
                features=row.to_dict(),
                prediction=int(nova_groups[position]),
                probability=float(probabilities[position]),
                model_version=model_version,
                product_code=product_codes[position],
            )
        except Exception:  # noqa: BLE001 - already logged by the store
            failures += 1

    if failures:
        logger.warning(
            "Batch persistence finished with %d/%d failed records.",
            failures,
            len(features),
        )


@app.post(
    "/v1/predict/batch",
    response_model=NovaBatchPredictResponse,
    status_code=status.HTTP_200_OK,
)
def predict_batch(
    request: NovaBatchPredictRequest,
    background_tasks: BackgroundTasks,
) -> NovaBatchPredictResponse:
    """Classify many products at once, reporting invalid rows instead of failing."""
    champion = _require_champion()
    model_version = str(MODEL_CONTAINER.get("model_version", "unknown"))

    valid_records, errors = _validate_records(request.records)
    if not valid_records:
        return NovaBatchPredictResponse(
            model_version=model_version, predictions=[], errors=errors
        )

    try:
        start_time = time.perf_counter()

        rows = [item.payload.model_dump(by_alias=True) for item in valid_records]
        features = pd.DataFrame(rows)[FEATURE_COLUMNS].astype(float)
        product_codes: list[Optional[str]] = [row.get("product_code") for row in rows]

        raw_labels, confidences = predict_with_confidence(champion, features)
        nova_groups = to_nova_groups(raw_labels)
        probabilities = np.round(validate_probabilities(confidences), 4)

        PREDICTION_LATENCY.observe(time.perf_counter() - start_time)
        groups, counts = np.unique(nova_groups, return_counts=True)
        for group, count in zip(groups, counts):
            PREDICTION_COUNTER.labels(nova_group=str(int(group))).inc(int(count))

        if request.persist:
            background_tasks.add_task(
                _persist_batch,
                features=features,
                nova_groups=nova_groups,
                probabilities=probabilities,
                product_codes=product_codes,
                model_version=model_version,
            )

        predictions = [
            BatchPrediction(
                index=item.index,
                product_code=product_codes[position],
                nova_group=int(nova_groups[position]),
                probability=float(probabilities[position]),
            )
            for position, item in enumerate(valid_records)
        ]

        return NovaBatchPredictResponse(
            model_version=model_version,
            predictions=predictions,
            errors=errors,
        )

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Batch inference execution error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch inference failed.",
        ) from exc
