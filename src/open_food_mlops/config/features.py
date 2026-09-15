"""Canonical dataset schema and feature configuration definitions."""

from __future__ import annotations

from typing import Final

TARGET_COLUMN: Final[str] = "nova_group"

CORE_NUTRITIONAL_VALUES = [
    "added-sugars_100g",
    "fruits-vegetables-legumes_100g",
    "energy_100g",
    "energy-kcal_100g",
    "proteins_100g",
    "carbohydrates_100g",
    "sugars_100g",
    "fat_100g",
    "saturated-fat_100g",
    "fiber_100g",
    "sodium_100g",
    "salt_100g",
    "water_100g",
]

ADDITIVES = [
    "additives_n",
    "additives_tags",
]

INGREDIENT_INFORMATION = [
    "ingredients_text",
]

CATEGORY_AND_DESCRIPTIVE_TAGS = [
    "main_category",
    "categories_tags",
    "pnns_groups_1",
    "pnns_groups_2",
]


#RAW_FEATURES: Final[list[str]] = CORE_NUTRITIONAL_VALUES + ADDITIVES + INGREDIENT_INFORMATION + CATEGORY_AND_DESCRIPTIVE_TAGS
RAW_FEATURES: Final[list[str]] = CORE_NUTRITIONAL_VALUES + ["additives_n"]

FEATURE_COLUMNS: Final[list[str]] = [f for f in RAW_FEATURES if f != TARGET_COLUMN]
