"""Unit tests for model class registration and retrieval."""

from __future__ import annotations

from typing import Any, Self

import pandas as pd
import pytest

from open_food_mlops.models.base import BaseModel
from open_food_mlops.models.registry import (
    _MODEL_REGISTRY,
    get_model_class,
    register,
    registered_models,
)
from open_food_mlops.models.specs import SearchSpace


class MockDummyModel(BaseModel):
    """Mock model class for testing registry mechanics."""

    model_name = "mock_dummy"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        return pd.Series([1] * len(X))

    @classmethod
    def get_search_space(cls) -> SearchSpace:
        return {}

    def _save(self, path: Any) -> None:
        pass

    @classmethod
    def _load(cls, path: Any) -> Self:
        return cls()


@pytest.fixture(autouse=True)
def clean_registry():
    """Ensure registry state is restored after each test run."""
    backup = dict(_MODEL_REGISTRY)
    yield
    _MODEL_REGISTRY.clear()
    _MODEL_REGISTRY.update(backup)


def test_register_and_get_model() -> None:
    """Verify successful model registration and normalized retrieval."""
    register("mock_dummy")(MockDummyModel)

    retrieved = get_model_class("MOCK_DUMMY")
    assert retrieved is MockDummyModel
    assert "mock_dummy" in list(registered_models())


def test_register_empty_name_raises() -> None:
    """Verify error on registering an empty or whitespace-only name."""
    with pytest.raises(ValueError, match="cannot be empty"):
        register("   ")(MockDummyModel)


def test_register_duplicate_name_raises() -> None:
    """Verify error when registering an already occupied model name."""
    register("mock_dummy")(MockDummyModel)
    with pytest.raises(ValueError, match="already registered"):
        register("mock_dummy")(MockDummyModel)


def test_register_non_basemodel_raises() -> None:
    """Verify error when attempting to register a class not inheriting from BaseModel."""
    class NonModel:
        pass

    with pytest.raises(TypeError, match="must inherit from BaseModel"):
        register("invalid")(NonModel)  # type: ignore[arg-type]


def test_get_unregistered_model_raises() -> None:
    """Verify KeyError with clear message when requesting an unknown model."""
    with pytest.raises(KeyError, match="Model 'unknown_model' is not registered"):
        get_model_class("unknown_model")