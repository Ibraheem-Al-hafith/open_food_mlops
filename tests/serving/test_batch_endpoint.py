from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from open_food_mlops.config.features import FEATURE_COLUMNS
from serving import app as app_module
from serving.prediction_store import PredictionStore


class FakeModel:
    """Predicts raw class 3 (NOVA 4) when additives_n >= 3, else raw class 0 (NOVA 1)."""

    classes_ = np.array([0, 1, 2, 3])

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        probabilities = np.full((len(frame), 4), 0.1)
        winners = np.where(frame["additives_n"] >= 3, 3, 0)
        probabilities[np.arange(len(frame)), winners] = 0.7
        return probabilities


def _record(**overrides: Any) -> dict[str, Any]:
    record: dict[str, Any] = {column: 1.0 for column in FEATURE_COLUMNS}
    record["fruits-vegetables-legumes_100g"] = 50.0
    record.update(overrides)
    return record


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch, tmp_path) -> TestClient:
    monkeypatch.setitem(app_module.MODEL_CONTAINER, "champion", FakeModel())
    monkeypatch.setitem(app_module.MODEL_CONTAINER, "model_version", "production")
    monkeypatch.setattr(app_module, "prediction_store", PredictionStore(tmp_path / "p.db"))
    return TestClient(app_module.app)


def test_batch_reports_valid_and_invalid_rows(client: TestClient) -> None:
    records = [
        _record(additives_n=0, product_code="a"),
        _record(additives_n=5, product_code="b"),
        _record(fat_100g=-1),
    ]

    body = client.post("/v1/predict/batch", json={"records": records}).json()

    assert body["model_version"] == "production"
    assert [(p["index"], p["product_code"], p["nova_group"]) for p in body["predictions"]] == [
        (0, "a", 1), (1, "b", 4),
    ]
    assert [e["index"] for e in body["errors"]] == [2]
    assert "fat_100g" in body["errors"][0]["detail"]


def test_batch_matches_single_prediction(client: TestClient) -> None:
    record = _record(additives_n=5)
    single = client.post("/v1/predict", json=record).json()
    batch = client.post("/v1/predict/batch", json={"records": [record]}).json()["predictions"][0]
    assert (single["nova_group"], single["probability"]) == (batch["nova_group"], batch["probability"])


def test_batch_only_persists_when_requested(client: TestClient) -> None:
    client.post("/v1/predict/batch", json={"records": [_record()]})
    assert app_module.prediction_store.read_predictions_dataframe().empty

    client.post("/v1/predict/batch", json={"records": [_record(product_code="x")], "persist": True})
    stored = app_module.prediction_store.read_predictions_dataframe()
    assert stored["product_code"].tolist() == ["x"] and stored["model_version"].tolist() == ["production"]


def test_batch_all_invalid_returns_errors_only(client: TestClient) -> None:
    body = client.post("/v1/predict/batch", json={"records": [{"nope": 1}]}).json()
    assert body["predictions"] == [] and len(body["errors"]) == 1


def test_batch_without_model_returns_503(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(app_module.MODEL_CONTAINER, "champion", None)
    assert client.post("/v1/predict/batch", json={"records": [_record()]}).status_code == 503


def test_batch_rejects_empty_request(client: TestClient) -> None:
    assert client.post("/v1/predict/batch", json={"records": []}).status_code == 422
