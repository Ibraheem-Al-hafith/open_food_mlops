"""Canonical dataset schema and feature configuration definitions."""

from __future__ import annotations

from typing import Final

TARGET_COLUMN: Final[str] = "nova_group"

RAW_FEATURES: Final[list[str]] = [
    "added-sugars_100g",
    "fat_100g",
    "proteins_100g",
    "fruits-vegetables-legumes_100g",
    "sodium_100g",
    "salt_100g",
    "energy-kcal_100g",
    "carbohydrates_100g",
    "water_100g",
]

FEATURE_COLUMNS: Final[list[str]] = [f for f in RAW_FEATURES if f != TARGET_COLUMN]