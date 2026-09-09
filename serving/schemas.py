"""Pydantic schemas for the FastAPI inference service."""

import math
from typing import Optional, Any
from pydantic import BaseModel, ConfigDict, Field, field_validator


class NovaPredictRequest(BaseModel):
    """Schema for single-product feature input values matching DataConfig."""

    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    product_code: Optional[str] = Field(
        None, description="Unique product barcode identifier for operational tracking"
    )
    added_sugars_100g: float = Field(..., alias="added-sugars_100g", ge=0.0)
    fat_100g: float = Field(..., ge=0.0)
    proteins_100g: float = Field(..., ge=0.0)
    fruits_vegetables_legumes_100g: float = Field(
        ..., alias="fruits-vegetables-legumes_100g", ge=0.0, le=100.0
    )
    sodium_100g: float = Field(..., ge=0.0)
    salt_100g: float = Field(..., ge=0.0)
    energy_kcal_100g: float = Field(..., alias="energy-kcal_100g", ge=0.0)
    carbohydrates_100g: float = Field(..., ge=0.0)
    water_100g: float = Field(..., ge=0.0)

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

    nova_group: int = Field(..., description="Predicted NOVA group (1 to 4)")
    probability: float = Field(
        ..., description="Confidence probability for the predicted class"
    )