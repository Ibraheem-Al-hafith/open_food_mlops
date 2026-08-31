"""Unit tests verifying unified TabularDataProvider integrity."""

import pandas as pd
import pytest

from open_food_mlops.data.providers import (
    DataPartitionConfig,
    HoldoutTestConfig,
    TabularDataProvider,
    ValidationConfig,
    ValidationMethod,
)


@pytest.fixture
def synthetic_dataset(tmp_path):
    """Creates a temporary Parquet dataset file for testing."""
    file_path = tmp_path / "processed_data.parquet"
    df = pd.DataFrame(
        {
            "feature1": range(100),
            "feature2": range(100, 200),
            "nova_group": [1, 2, 3, 4] * 25,
        }
    )
    df.to_parquet(file_path, index=False)
    return str(file_path)


def test_holdout_test_set_isolation(synthetic_dataset):
    """Ensure test set is correctly isolated from cross-validation dev pool."""
    config = DataPartitionConfig(
        test=HoldoutTestConfig(test_size=0.2, random_state=42),
        validation=ValidationConfig(method=ValidationMethod.STRATIFIED_K_FOLD, n_splits=5),
    )

    provider = TabularDataProvider(
        data_path=synthetic_dataset, target_column="nova_group", config=config
    )

    X_test, y_test = provider.test_set
    assert len(X_test) == 20
    assert len(y_test) == 20

    splits = list(provider.get_splits())
    assert len(splits) == 5

    # Check first fold dev partition size (80 total dev rows split into 5 folds)
    # 80 / 5 = 16 validation samples per fold, 64 train samples
    assert len(splits[0].X_train) == 64
    assert len(splits[0].X_val) == 16


def test_invalid_target_raises_key_error(synthetic_dataset):
    """Verify missing target column raises explicit KeyError."""
    with pytest.raises(KeyError, match="Target column 'invalid_target' not found"):
        TabularDataProvider(data_path=synthetic_dataset, target_column="invalid_target")