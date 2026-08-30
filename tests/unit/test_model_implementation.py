"""Unit tests for built-in estimator wrappers (Decision Tree, RF, Logistic Regression, LightGBM, XGBoost)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from open_food_mlops.models.implementations.decision_tree import DecisionTreeModel
from open_food_mlops.models.implementations.lightgbm import LightGBMModel, lgb
from open_food_mlops.models.implementations.logistic_regression import LogisticRegressionModel
from open_food_mlops.models.implementations.random_forest import RandomForestModel
from open_food_mlops.models.implementations.xgboost import XGBoostModel, xgb


@pytest.mark.parametrize(
    "model_cls",
    [
        DecisionTreeModel,
        LogisticRegressionModel,
        RandomForestModel,
    ],
)
def test_standard_models_fit_predict_save_load(
    model_cls: type,
    synthetic_binary_data: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """Test full training, inference, and serialization lifecycle for standard models."""
    X, y = synthetic_binary_data
    model = model_cls()

    model.fit(X, y)
    assert model.is_fitted_

    preds = model.predict(X)
    assert isinstance(preds, pd.Series)
    assert len(preds) == len(X)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        assert isinstance(probs, pd.DataFrame)
        assert probs.shape == (len(X), 2)

    save_path = tmp_path / model_cls.model_name
    model.save(save_path)

    restored = model_cls.load(save_path)
    assert restored.is_fitted_
    restored_preds = restored.predict(X)
    pd.testing.assert_series_equal(preds, restored_preds)


@pytest.mark.skipif(lgb is None, reason="lightgbm is not installed")
def test_lightgbm_model_lifecycle(
    synthetic_binary_data: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """Test LightGBM adapter lifecycle when installed."""
    X, y = synthetic_binary_data
    model = LightGBMModel({"n_estimators": 10})
    model.fit(X, y)

    preds = model.predict(X)
    probs = model.predict_proba(X)
    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)

    save_path = tmp_path / "lgb"
    model.save(save_path)
    restored = LightGBMModel.load(save_path)
    assert restored.is_fitted_


def test_lightgbm_import_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify LightGBMModel raises ImportError when lightgbm library is missing."""
    import open_food_mlops.models.implementations.lightgbm as lgb_mod

    monkeypatch.setattr(lgb_mod, "lgb", None)
    with pytest.raises(ImportError, match="LightGBM is not installed"):
        lgb_mod.LightGBMModel()


@pytest.mark.skipif(xgb is None, reason="xgboost is not installed")
def test_xgboost_model_lifecycle(
    synthetic_binary_data: tuple[pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    """Test XGBoost adapter lifecycle when installed."""
    X, y = synthetic_binary_data
    model = XGBoostModel({"n_estimators": 10})
    model.fit(X, y)

    preds = model.predict(X)
    probs = model.predict_proba(X)
    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)

    save_path = tmp_path / "xgb"
    model.save(save_path)
    restored = XGBoostModel.load(save_path)
    assert hasattr(restored.estimator_, "classes_")
    # or check values directly:
    assert (restored.estimator_.classes_ == model.estimator_.classes_).all()
    assert restored.is_fitted_


def test_xgboost_import_error_handling(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify XGBoostModel raises ImportError when xgboost library is missing."""
    import open_food_mlops.models.implementations.xgboost as xgb_mod

    monkeypatch.setattr(xgb_mod, "xgb", None)
    with pytest.raises(ImportError, match="XGBoost is not installed"):
        xgb_mod.XGBoostModel()