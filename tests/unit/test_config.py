"""Unit tests for Pydantic configuration schemas."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from open_food_mlops.models.config import ModelConfig, TuningConfig


def test_tuning_config_defaults() -> None:
    """Verify default values for TuningConfig."""
    config = TuningConfig()
    assert not config.enabled
    assert config.method == "optuna"
    assert config.trials == 50
    assert config.direction == "maximize"
    assert config.random_state == 42


def test_tuning_config_validation() -> None:
    """Verify strict validation and extra field rejection in TuningConfig."""
    with pytest.raises(ValidationError):
        TuningConfig(trials=0)

    with pytest.raises(ValidationError):
        TuningConfig(invalid_extra_field=True)  # type: ignore[call-arg]


def test_model_config_initialization() -> None:
    """Verify ModelConfig initialization and nesting."""
    config = ModelConfig(
        name="random_forest",
        params={"n_estimators": 200},
        tuning=TuningConfig(enabled=True, trials=10),
    )
    assert config.name == "random_forest"
    assert config.params["n_estimators"] == 200
    assert config.tuning.enabled
    assert config.tuning.trials == 10