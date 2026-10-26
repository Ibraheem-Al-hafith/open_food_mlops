from food_ui.consistency import check_plausibility
from food_ui.examples import EXAMPLE_PRODUCTS
from food_ui.features import FEATURE_KEYS


def _values(**overrides: float) -> dict[str, float]:
    base = dict(EXAMPLE_PRODUCTS["Wholemeal bread"])
    base.update({key.replace("__", "-"): value for key, value in overrides.items()})
    return base


def test_all_examples_are_plausible() -> None:
    for name, product in EXAMPLE_PRODUCTS.items():
        assert set(product) == set(FEATURE_KEYS), name
        assert check_plausibility(product) == [], name


def test_sugars_above_carbohydrates_is_flagged() -> None:
    warnings = check_plausibility(_values(sugars_100g=60, carbohydrates_100g=41))
    assert any("Sugars are higher" in w for w in warnings)


def test_energy_unit_mixup_is_flagged() -> None:
    warnings = check_plausibility(_values(energy_100g=240))  # kcal typed into the kJ field
    assert any("kJ and kcal" in w for w in warnings)


def test_salt_sodium_mismatch_is_flagged() -> None:
    warnings = check_plausibility(_values(salt_100g=3.0, sodium_100g=0.4))
    assert any("Salt and sodium" in w for w in warnings)


def test_zero_values_never_trigger_cross_checks() -> None:
    zeros = {key: 0.0 for key in FEATURE_KEYS}
    assert check_plausibility(zeros) == []
