"""The UI duplicates the feature list; this test fails if the model contract changes."""

from open_food_mlops.config.features import FEATURE_COLUMNS
from food_ui.features import FEATURE_KEYS


def test_ui_features_match_model_features() -> None:
    assert set(FEATURE_KEYS) == set(FEATURE_COLUMNS)
    assert len(FEATURE_KEYS) == len(set(FEATURE_KEYS))
