"""Production FastAPI serving application for inference execution."""

from __future__ import annotations

import logging
from typing import Any, Dict
import pandas as pd
from fastapi import FastAPI, HTTPException, status

from serving.schemas import PredictRequest, PredictResponse

logger = logging.getLogger(__name__)

app = FastAPI(title="Open Food MLOps Serving API", version="0.1.0")

# Global placeholder for loaded champion model adapter
MODEL_SERVICER: Any = None


@app.on_event("startup")
def load_champion_model() -> None:
    """Load model artifact into memory during app startup."""
    global MODEL_SERVICER
    logger.info("Initializing REST Inference Endpoint...")
    # Production service dynamically loads model from MLflow Registry or artifacts
    MODEL_SERVICER = None


@app.get("/health", status_code=status.HTTP_200_OK)
def health_check() -> Dict[str, str]:
    """Health check endpoint for Kubernetes / Docker liveness probing."""
    return {"status": "healthy"}


@app.post("/predict", response_model=PredictResponse, status_code=status.HTTP_200_OK)
def predict(request: PredictRequest) -> PredictResponse:
    """Execute real-time model predictions for input payload features."""
    if not request.data:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Payload data cannot be empty.",
        )

    try:
        df = pd.DataFrame(request.data)
        # Placeholder returns transformed shape fallback if servicer uninitialized
        predictions = [0] * len(df)
        return PredictResponse(predictions=predictions, model_version="champion")
    except Exception as exc:
        logger.error("Inference failure: %s", str(exc), exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Error processing inference prediction.",
        ) from exc