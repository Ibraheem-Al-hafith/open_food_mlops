"""Unit tests for the create_model factory function."""

from __future__ import annotations

import pytest

from open_food_mlops.models.config import ModelConfig
from open_food_mlops.models.factory import create_model
from open_food_mlops.models.implementations.decision_tree import DecisionTreeModel


def test_create_model_success() -> None:
    """Verify factory builds and parameter overrides on registered models."""
    config = ModelConfig(
        name="decision_tree",
        params={"max_depth": 5, "criterion": "gini"},
    )

    model = create_model(config, criterion="entropy")
    assert isinstance(model, DecisionTreeModel)
    assert model.config["max_depth"] == 5
    assert model.config["criterion"] == "entropy"


def test_create_model_unregistered_raises() -> None:
    """Verify factory raises KeyError for unregistered model names."""
    config = ModelConfig(name="non_existent_model")
    with pytest.raises(KeyError):
        create_model(config)