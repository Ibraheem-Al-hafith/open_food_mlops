"""Pydantic schemas for the FastAPI inference service."""

from typing import Any, Dict, List
from pydantic import BaseModel


class PredictRequest(BaseModel):
    """Schema for batch inference request payloads."""

    data: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    """Schema for prediction output responses."""

    predictions: List[Any]
    model_version: str = "champion"