from __future__ import annotations

import httpx
import pytest

from food_ui.client import ApiConnectionError, ApiError, ModelUnavailableError, PredictionClient


def _client(handler) -> PredictionClient:
    return PredictionClient("http://api", transport=httpx.MockTransport(handler))


def test_predict_sends_product_code_and_parses_result() -> None:
    seen: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(httpx.Response(200, content=request.content).json())
        return httpx.Response(200, json={"nova_group": 3, "probability": 0.8})

    result = _client(handler).predict({"fat_100g": 1.0}, product_code="abc")

    assert (result.nova_group, result.probability) == (3, 0.8)
    assert seen["product_code"] == "abc"


def test_503_maps_to_model_unavailable() -> None:
    with pytest.raises(ModelUnavailableError):
        _client(lambda r: httpx.Response(503, json={"detail": "Model is unavailable."})).predict({})


def test_422_surfaces_field_message() -> None:
    body = {"detail": [{"loc": ["body", "fat_100g"], "msg": "must be >= 0"}]}
    with pytest.raises(ApiError, match="fat_100g: must be >= 0"):
        _client(lambda r: httpx.Response(422, json=body)).predict({})


def test_connection_failure_maps_to_connection_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    with pytest.raises(ApiConnectionError):
        _client(handler).predict({})


def test_health_never_raises() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    assert _client(handler).health().healthy is False


def test_health_reports_version() -> None:
    payload = {"status": "healthy", "model_loaded": True, "model_version": "production"}
    status = _client(lambda r: httpx.Response(200, json=payload)).health()
    assert status.healthy and status.model_version == "production"


def test_predict_batch_parses_payload() -> None:
    payload = {
        "model_version": "production",
        "predictions": [{"index": 0, "product_code": None, "nova_group": 2, "probability": 0.7}],
        "errors": [{"index": 1, "detail": "bad"}],
    }
    result = _client(lambda r: httpx.Response(200, json=payload)).predict_batch([{}, {}])
    assert result.predictions[0].nova_group == 2 and result.errors[0].detail == "bad"
