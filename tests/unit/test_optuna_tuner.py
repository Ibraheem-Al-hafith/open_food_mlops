"""Unit tests for Optuna hyperparameter optimization engine."""

from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from open_food_mlops.models.implementations.decision_tree import DecisionTreeModel
from open_food_mlops.models.tuning.optuna_tuner import OptunaTuner, optuna


@pytest.mark.skipif(optuna is None, reason="optuna package is not installed")
def test_optuna_tuner_optimization_loop(
    synthetic_binary_data: tuple[pd.DataFrame, pd.Series],
) -> None:
    """Verify end-to-end Optuna hyperparameter search."""
    X, y = synthetic_binary_data
    model_cls = DecisionTreeModel
    search_space = model_cls.get_search_space()

    tuner = OptunaTuner(
        model_class=model_cls,
        search_space=search_space,
        n_trials=3,
        direction="maximize",
        random_state=42,
    )

    def objective(params: dict[str, Any]) -> float:
        model = model_cls(params)
        model.fit(X, y)
        preds = model.predict(X)
        return float((preds == y).mean())

    result = tuner.optimize(objective)

    assert result.n_trials == 3
    assert isinstance(result.best_score, float)
    assert isinstance(result.best_params, dict)
    assert "max_depth" in result.best_params


def test_optuna_tuner_validation() -> None:
    """Verify input validation for invalid optimization directions and trial budgets."""
    if optuna is None:
        pytest.skip("Optuna is not installed")

    with pytest.raises(ValueError, match="Invalid optimization direction"):
        OptunaTuner(
            model_class=DecisionTreeModel,
            search_space={},
            direction="invalid",
        )

    with pytest.raises(ValueError, match="n_trials must be a positive integer"):
        OptunaTuner(
            model_class=DecisionTreeModel,
            search_space={},
            n_trials=0,
        )


def test_optuna_tuner_missing_dependency(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify OptunaTuner raises ImportError when optuna package is missing."""
    import open_food_mlops.models.tuning.optuna_tuner as optuna_mod

    monkeypatch.setattr(optuna_mod, "optuna", None)

    with pytest.raises(ImportError, match="Optuna package is not installed"):
        optuna_mod.OptunaTuner(
            model_class=DecisionTreeModel,
            search_space={},
        )