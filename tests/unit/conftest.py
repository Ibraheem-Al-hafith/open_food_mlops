"""Shared test fixtures for the model subsystem test suite."""

from __future__ import annotations

import pandas as pd
import pytest
from sklearn.datasets import make_classification


@pytest.fixture
def synthetic_binary_data() -> tuple[pd.DataFrame, pd.Series]:
    """Generate a reproducible binary classification dataset.

    Returns:
        Tuple containing feature DataFrame (X) and target Series (y).
    """
    X_raw, y_raw = make_classification(
        n_samples=100,
        n_features=5,
        n_informative=3,
        n_redundant=1,
        random_state=42,
    )
    feature_names = [f"feature_{i}" for i in range(5)]
    X = pd.DataFrame(X_raw, columns=feature_names)
    y = pd.Series(y_raw, name="target")
    return X, y