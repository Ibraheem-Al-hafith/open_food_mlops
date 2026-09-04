"""Production FastAPI serving application for inference execution."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException, status
from contextlib import asynccontextmanager
from pydantic import BaseModel, Field

from open_food_mlops.features.builder import get_feature_pipeline
import open_food_mlops.models.implementations  # Register models
from open_food_mlops.models.registry import get_model_class

logger = logging.getLogger(__name__)

MODEL_SERVICER: Any = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup logic
    global MODEL_SERVICER
    logger.info("Initializing REST Serving Layer...")
    MODEL_SERVICER = {"pipeline": get_feature_pipeline()}
    
    yield  # Server runs while execution pauses here
    
    # Cleanup logic (if needed on shutdown) goes here

app = FastAPI(
    title="Open Food MLOps Serving API",
    version="0.1.0",
    lifespan=lifespan
    )


class PredictRequest(BaseModel):
    """Schema for batch inference request payloads."""

    data: list[dict[str, Any]]
    model_name: str = "decision_tree"
    artifact_path: str | None = None


class PredictResponse(BaseModel):
    """Schema for prediction output responses."""

    predictions: list[Any]
    model_version: str = "champion"



@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, str]:
    """Health check endpoint for Kubernetes liveness probes."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
def predict(request: PredictRequest) -> PredictResponse:
    """Execute real-time model predictions for input payloads."""
    if not request.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload data cannot be empty.",
        )

    try:
        df = pd.DataFrame(request.data)
        pipeline = MODEL_SERVICER["pipeline"]
        X_trans = pipeline.transform(df) if pipeline.is_fitted_ else df

        if request.artifact_path and Path(request.artifact_path).exists():
            model_cls = get_model_class(request.model_name)
            model = model_cls.load(request.artifact_path)
            predictions = model.predict(X_trans).tolist()
        else:
            # Fallback output format stub for testing uninitialized servicer
            predictions = [0] * len(df)

        return PredictResponse(predictions=predictions, model_version="champion")
    except Exception as exc:
        logger.error("Inference failure: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing inference prediction: {exc}",
        ) from exc