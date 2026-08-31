"""Unit tests verifying data providers, evaluators, and model selection engine."""

import pandas as pd
import pytest
#from open_food_mlops.data.providers import TabularDataProvider, DataPartitionConfig, HoldoutTestConfig, ValidationConfig, ValidationMethod
from open_food_mlops.evaluation.evaluator import Evaluator
from open_food_mlops.experiments.selection import (
    CandidateResult,
    ModelSelectionEngine,
)



def test_evaluator_metrics():
    """Verify classification metric outputs."""
    evaluator = Evaluator(primary_metric="macro_f1")
    y_true = pd.Series([0, 1, 0, 1])
    y_pred = pd.Series([0, 1, 0, 0])

    res = evaluator.evaluate(y_true, y_pred)
    assert "accuracy" in res.metrics
    assert res.metrics["accuracy"] == 0.75
    assert res.primary_score == res.metrics["macro_f1"]


def test_selection_engine_quality_gates():
    """Verify quality gate rejection and champion ranking logic."""
    engine = ModelSelectionEngine(
        primary_metric="macro_f1", direction="maximize", gates={"macro_f1": 0.8}
    )

    c1 = CandidateResult(
        candidate_id="1",
        model_name="m1",
        metrics={"macro_f1": 0.75},
        params={},
        artifact_path="",
    )
    c2 = CandidateResult(
        candidate_id="2",
        model_name="m2",
        metrics={"macro_f1": 0.90},
        params={},
        artifact_path="",
    )

    result = engine.select_champion([c1, c2])

    assert len(result.rejected_candidates) == 1
    assert result.rejected_candidates[0].candidate_id == "1"
    assert result.champion is not None
    assert result.champion.candidate_id == "2"