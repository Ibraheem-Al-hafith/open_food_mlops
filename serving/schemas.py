"""Pydantic schemas for the FastAPI inference service."""

import math
from typing import Any, Final, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Hard upper bound of records accepted by a single batch request.
MAX_BATCH_SIZE: Final[int] = 5000


class NovaPredictRequest(BaseModel):
    """Schema for single-product feature input values matching DataConfig."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    product_code: Optional[str] = Field(
        None, alias="product_code", description="Unique product barcode identifier for operational tracking"
    )
    energy_100g: float = Field(..., ge=0.0)
    added_sugars_100g: float = Field(..., alias="added-sugars_100g", ge=0.0)
    saturated_fat_100g: float = Field(..., alias="saturated-fat_100g", ge=0.0)
    fat_100g: float = Field(..., ge=0.0)
    fiber_100g: float = Field(..., ge=0.0)
    sugars_100g: float = Field(..., ge=0.0)
    proteins_100g: float = Field(..., ge=0.0)
    fruits_vegetables_legumes_100g: float = Field(
        ..., alias="fruits-vegetables-legumes_100g", ge=0.0, le=100.0
    )
    sodium_100g: float = Field(..., ge=0.0)
    salt_100g: float = Field(..., ge=0.0)
    energy_kcal_100g: float = Field(..., alias="energy-kcal_100g", ge=0.0)
    carbohydrates_100g: float = Field(..., ge=0.0)
    water_100g: float = Field(..., ge=0.0)
    additives_n: float = Field(..., ge=0.0)

    @field_validator("*", mode="before")
    @classmethod
    def validate_finite_numbers(cls, value: float, info: Any) -> float:
        """Reject non-finite values (NaN, inf, -inf) across float features."""
        if info.field_name == "product_code" or value is None:
            return value
        if isinstance(value, (int, float)):
            if not math.isfinite(value):
                raise ValueError(f"Field '{info.field_name}' must contain a finite numeric value.")
        return value


class NovaPredictResponse(BaseModel):
    """Schema for NOVA score prediction output."""

    model_config = ConfigDict(extra="forbid")

    nova_group: int = Field(
        ...,
        ge=1,
        le=4,
        description="Predicted NOVA group (1 to 4)",
    )

    probability: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence probability for the predicted class",
    )

    @field_validator("probability", mode="before")
    @classmethod
    def validate_probability(cls, value: float) -> float:
        """Reject non-finite probability values."""
        if isinstance(value, (int, float)) and not math.isfinite(value):
            raise ValueError("Probability must be a finite numeric value.")
        return value


class NovaBatchPredictRequest(BaseModel):
    """Schema for a batch of products. Each record is validated individually."""

    model_config = ConfigDict(extra="forbid")

    records: list[dict[str, Any]] = Field(
        ...,
        min_length=1,
        max_length=MAX_BATCH_SIZE,
        description="Raw feature records, validated one by one against NovaPredictRequest.",
    )
    persist: bool = Field(
        False,
        description="Store predictions in the production prediction store (feeds monitoring).",
    )


class BatchPrediction(BaseModel):
    """Prediction for one successfully validated batch record."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., ge=0, description="Position of the record in the request.")
    product_code: Optional[str] = None
    nova_group: int = Field(..., ge=1, le=4)
    probability: float = Field(..., ge=0.0, le=1.0)


class BatchRowError(BaseModel):
    """Validation failure for one batch record."""

    model_config = ConfigDict(extra="forbid")

    index: int = Field(..., ge=0, description="Position of the record in the request.")
    detail: str


class NovaBatchPredictResponse(BaseModel):
    """Batch result: valid rows are predicted, invalid rows are reported."""

    model_config = ConfigDict(extra="forbid")

    model_version: str
    predictions: list[BatchPrediction]
    errors: list[BatchRowError]
