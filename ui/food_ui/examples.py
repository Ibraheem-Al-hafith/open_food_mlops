"""Approximate example products used to pre-fill the online form."""

from __future__ import annotations

from typing import Final, Mapping

# Values are per 100 g and only indicative; they exist to demo the form.
EXAMPLE_PRODUCTS: Final[Mapping[str, Mapping[str, float]]] = {
    "Fresh apple": {
        "energy_100g": 218, "energy-kcal_100g": 52, "fat_100g": 0.2, "saturated-fat_100g": 0.03,
        "carbohydrates_100g": 13.8, "sugars_100g": 10.4, "added-sugars_100g": 0, "fiber_100g": 2.4,
        "proteins_100g": 0.3, "salt_100g": 0, "sodium_100g": 0, "water_100g": 85.6,
        "fruits-vegetables-legumes_100g": 100, "additives_n": 0,
    },
    "Olive oil": {
        "energy_100g": 3700, "energy-kcal_100g": 884, "fat_100g": 100, "saturated-fat_100g": 14,
        "carbohydrates_100g": 0, "sugars_100g": 0, "added-sugars_100g": 0, "fiber_100g": 0,
        "proteins_100g": 0, "salt_100g": 0, "sodium_100g": 0, "water_100g": 0,
        "fruits-vegetables-legumes_100g": 0, "additives_n": 0,
    },
    "Wholemeal bread": {
        "energy_100g": 1000, "energy-kcal_100g": 240, "fat_100g": 3, "saturated-fat_100g": 0.7,
        "carbohydrates_100g": 41, "sugars_100g": 3, "added-sugars_100g": 2, "fiber_100g": 6.5,
        "proteins_100g": 9, "salt_100g": 1.0, "sodium_100g": 0.4, "water_100g": 38,
        "fruits-vegetables-legumes_100g": 0, "additives_n": 1,
    },
    "Potato crisps": {
        "energy_100g": 2250, "energy-kcal_100g": 540, "fat_100g": 33, "saturated-fat_100g": 3.5,
        "carbohydrates_100g": 50, "sugars_100g": 0.6, "added-sugars_100g": 0, "fiber_100g": 4.5,
        "proteins_100g": 6, "salt_100g": 1.3, "sodium_100g": 0.52, "water_100g": 2,
        "fruits-vegetables-legumes_100g": 0, "additives_n": 2,
    },
    "Cola soft drink": {
        "energy_100g": 180, "energy-kcal_100g": 42, "fat_100g": 0, "saturated-fat_100g": 0,
        "carbohydrates_100g": 10.6, "sugars_100g": 10.6, "added-sugars_100g": 10.6, "fiber_100g": 0,
        "proteins_100g": 0, "salt_100g": 0, "sodium_100g": 0, "water_100g": 89,
        "fruits-vegetables-legumes_100g": 0, "additives_n": 3,
    },
}
