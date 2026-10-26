"""Feature catalogue describing every model input shown in the UI.

The keys are the exact column names accepted by the serving API. They are
duplicated here on purpose so the UI image does not need the ML package; a
contract test (``tests/test_ui_feature_contract.py (in the main project)``) keeps both in sync.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final, Optional

PRODUCT_CODE_COLUMN: Final[str] = "product_code"


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """One numeric model input."""

    key: str
    label: str
    unit: str
    group: str
    maximum: Optional[float] = None
    help_text: str = ""


FEATURES: Final[tuple[FeatureSpec, ...]] = (
    FeatureSpec("energy_100g", "Energy", "kJ", "Energy"),
    FeatureSpec("energy-kcal_100g", "Energy", "kcal", "Energy"),
    FeatureSpec("fat_100g", "Fat", "g", "Fat"),
    FeatureSpec("saturated-fat_100g", "Saturated fat", "g", "Fat", help_text="Part of total fat."),
    FeatureSpec("carbohydrates_100g", "Carbohydrates", "g", "Carbohydrates"),
    FeatureSpec("sugars_100g", "Sugars", "g", "Carbohydrates", help_text="Part of carbohydrates."),
    FeatureSpec(
        "added-sugars_100g", "Added sugars", "g", "Carbohydrates", help_text="Part of sugars."
    ),
    FeatureSpec("fiber_100g", "Fibre", "g", "Carbohydrates"),
    FeatureSpec("proteins_100g", "Protein", "g", "Protein, salt and water"),
    FeatureSpec("salt_100g", "Salt", "g", "Protein, salt and water"),
    FeatureSpec(
        "sodium_100g", "Sodium", "g", "Protein, salt and water", help_text="Salt is about 2.5 x sodium."
    ),
    FeatureSpec("water_100g", "Water", "g", "Protein, salt and water"),
    FeatureSpec(
        "fruits-vegetables-legumes_100g",
        "Fruits, vegetables, legumes",
        "%",
        "Protein, salt and water",
        maximum=100.0,
        help_text="Estimated share of the product.",
    ),
    FeatureSpec("additives_n", "Additives", "count", "Additives", help_text="Number of additives."),
)

FEATURE_KEYS: Final[tuple[str, ...]] = tuple(spec.key for spec in FEATURES)

FEATURE_GROUPS: Final[tuple[str, ...]] = tuple(dict.fromkeys(spec.group for spec in FEATURES))


def component_label(spec: FeatureSpec) -> str:
    """Return the label shown on the input widget, e.g. ``Energy (kJ)``."""
    return f"{spec.label} ({spec.unit})"
