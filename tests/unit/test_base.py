"""Unit tests for BaseModel core behaviors and contract enforcement."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Self

import pandas as pd
import pytest

from open_food_mlops.models.base import BaseModel
from open_food_mlops.models.specs import SearchSpace


class ConcreteModel(BaseModel):
    """Concrete subclass of BaseModel for checking shared methods."""

    model_name = "concrete_model"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> Self:
        self.is_fitted_ = True
        return self

    def predict(self, X: pd.DataFrame) -> Any:
        self._check_is_fitted()
        return pd.Series([0] * len(X), index=X.index)

    @classmethod
    def get_search_space(cls) -> SearchSpace:
        return {}

    def _save(self, path: Path) -> None:
        (path / "dummy.txt").write_text("saved")

    @classmethod
    def _load(cls, path: Path) -> Self:
        model = cls()
        model.is_fitted_ = True
        return model


def test_base_model_params_management() -> None:
    """Verify default parameters, configuration getting, and updating."""
    model = ConcreteModel({"a": 1})
    assert model.get_params() == {"a": 1}

    model.set_params(b=2)
    assert model.get_params() == {"a": 1, "b": 2}


def test_check_is_fitted_raises() -> None:
    """Verify error when operations requiring a fitted model are invoked prematurely."""
    model = ConcreteModel()
    X = pd.DataFrame({"col": [1, 2]})

    with pytest.raises(RuntimeError, match="has not been fitted"):
        model.predict(X)

    with pytest.raises(RuntimeError, match="has not been fitted"):
        model.save("tmp_path")


def test_save_and_load(tmp_path: Path) -> None:
    """Verify persistence directory creation and save/load workflow."""
    model = ConcreteModel()
    X = pd.DataFrame({"col": [1, 2]})
    model.fit(X, pd.Series([0, 1]))

    save_dir = tmp_path / "model_out"
    model.save(save_dir)

    assert (save_dir / "dummy.txt").exists()

    restored = ConcreteModel.load(save_dir)
    assert restored.is_fitted_