"""Integration test verifying full Orchestrator pipeline run across models."""

import pandas as pd
import pytest
import yaml
from open_food_mlops.config.schemas import ExperimentPlan
from open_food_mlops.experiments.orchestrator import ExperimentOrchestrator


def test_end_to_end_orchestrator_flow(tmp_path):
    """Executes full experiment pipeline using a synthetic CSV dataset."""
    # 1. Generate synthetic dataset
    csv_file = tmp_path / "food_data.csv"
    data = pd.DataFrame(
        {
            "calories": [100, 200, 150, 300, 250, 110, 210, 160, 310, 260],
            "fat": [5, 10, 7, 15, 12, 6, 11, 8, 16, 13],
            "target": [0, 1, 0, 1, 0, 0, 1, 0, 1, 0],
        }
    )
    data.to_csv(csv_file, index=False)

    # 2. Build mock ExperimentPlan
    raw_plan = {
        "name": "integration_test_exp",
        "data": {
            "provider": "tabular",
            "data_path": str(csv_file),
            "target_column": "target",
            "n_splits": 2,
        },
        "models": [
            {
                "name": "decision_tree",
                "enabled": True,
                "params": {"max_depth": 3},
                "tuning": {"enabled": False},
            },
            {
                "name": "logistic_regression",
                "enabled": True,
                "params": {"max_iter": 100},
                "tuning": {"enabled": False},
            },
        ],
        "selection": {
            "primary_metric": "macro_f1",
            "direction": "maximize",
            "gates": {"accuracy": 0.50},
        },
        "tracking": {
            "backend": "mlflow",
            "tracking_uri": f"sqlite:///{tmp_path}/mlflow.db",
            "experiment_name": "integration_tests",
        },
    }

    plan = ExperimentPlan(**raw_plan)

    # 3. Execute Orchestrator
    orchestrator = ExperimentOrchestrator(plan)
    selection_result = orchestrator.run()

    # 4. Assertions
    assert selection_result.champion is not None
    assert selection_result.champion.model_name in [
        "decision_tree",
        "logistic_regression",
    ]
    assert "accuracy" in selection_result.champion.metrics