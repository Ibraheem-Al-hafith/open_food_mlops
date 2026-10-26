"""Advisory plausibility checks for nutrition values entered by hand.

These checks never block a prediction; the API remains the source of truth for
validation. They only catch typical data-entry mistakes (swapped fields,
kJ/kcal mix-ups, impossible totals).
"""

from __future__ import annotations

from typing import Callable, Final, Mapping, Optional

KCAL_TO_KJ: Final[float] = 4.184
ENERGY_TOLERANCE: Final[float] = 0.15
SALT_PER_SODIUM: Final[float] = 2.5
SALT_TOLERANCE: Final[float] = 0.25
SALT_MIN_DIFFERENCE_G: Final[float] = 0.05
MAX_KCAL_PER_100G: Final[float] = 900.0
MAX_TOTAL_MASS_G: Final[float] = 105.0
PART_TOLERANCE_G: Final[float] = 0.05

Values = Mapping[str, float]
Rule = Callable[[Values], Optional[str]]


def _energy_units_match(v: Values) -> Optional[str]:
    kj, kcal = v["energy_100g"], v["energy-kcal_100g"]
    if kj > 0 and kcal > 0 and abs(kcal * KCAL_TO_KJ - kj) > ENERGY_TOLERANCE * kj:
        return "Energy in kJ and kcal do not match (1 kcal is about 4.2 kJ)."
    return None


def _energy_upper_bound(v: Values) -> Optional[str]:
    if v["energy-kcal_100g"] > MAX_KCAL_PER_100G:
        return "Energy is above 900 kcal per 100 g, the value of pure fat."
    return None


def _sugars_within_carbohydrates(v: Values) -> Optional[str]:
    if v["sugars_100g"] > v["carbohydrates_100g"] + PART_TOLERANCE_G:
        return "Sugars are higher than total carbohydrates."
    return None


def _added_within_sugars(v: Values) -> Optional[str]:
    if v["added-sugars_100g"] > v["sugars_100g"] + PART_TOLERANCE_G:
        return "Added sugars are higher than total sugars."
    return None


def _saturated_within_fat(v: Values) -> Optional[str]:
    if v["saturated-fat_100g"] > v["fat_100g"] + PART_TOLERANCE_G:
        return "Saturated fat is higher than total fat."
    return None


def _salt_matches_sodium(v: Values) -> Optional[str]:
    salt, sodium = v["salt_100g"], v["sodium_100g"]
    if salt <= 0 or sodium <= 0:
        return None
    expected = sodium * SALT_PER_SODIUM
    difference = abs(salt - expected)
    if difference > SALT_MIN_DIFFERENCE_G and difference > SALT_TOLERANCE * max(salt, expected):
        return "Salt and sodium do not match (salt is about 2.5 x sodium)."
    return None


def _total_mass_plausible(v: Values) -> Optional[str]:
    total = sum(
        v[key]
        for key in ("fat_100g", "carbohydrates_100g", "proteins_100g",
                    "fiber_100g", "water_100g", "salt_100g")
    )
    if total > MAX_TOTAL_MASS_G:
        return f"Fat, carbohydrates, protein, fibre, water and salt add up to {total:.0f} g per 100 g."
    return None


RULES: Final[tuple[Rule, ...]] = (
    _energy_units_match,
    _energy_upper_bound,
    _sugars_within_carbohydrates,
    _added_within_sugars,
    _saturated_within_fat,
    _salt_matches_sodium,
    _total_mass_plausible,
)


def check_plausibility(values: Values) -> list[str]:
    """Return a human readable warning for every violated rule."""
    return [message for rule in RULES if (message := rule(values)) is not None]
