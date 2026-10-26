"""Typed HTTP client for the serving API."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

import httpx


class ApiError(Exception):
    """The serving API answered with an error or an unexpected payload."""


class ApiConnectionError(ApiError):
    """The serving API could not be reached in time."""


class ModelUnavailableError(ApiError):
    """The serving API is up but has no model loaded (HTTP 503)."""


@dataclass(frozen=True, slots=True)
class HealthStatus:
    """Readiness of the serving API."""

    healthy: bool
    model_version: Optional[str] = None


@dataclass(frozen=True, slots=True)
class PredictionResult:
    """Outcome of a single online prediction."""

    nova_group: int
    probability: float


@dataclass(frozen=True, slots=True)
class BatchPrediction:
    """Prediction for one record of a batch call (``index`` is chunk-relative)."""

    index: int
    product_code: Optional[str]
    nova_group: int
    probability: float


@dataclass(frozen=True, slots=True)
class BatchRowFailure:
    """Record the API refused to predict (``index`` is chunk-relative)."""

    index: int
    detail: str


@dataclass(frozen=True, slots=True)
class BatchResult:
    """Outcome of one batch call."""

    model_version: str
    predictions: tuple[BatchPrediction, ...]
    errors: tuple[BatchRowFailure, ...]


class PredictionClient:
    """Thin wrapper around ``/health``, ``/v1/predict`` and ``/v1/predict/batch``."""

    def __init__(
        self,
        base_url: str,
        timeout_s: float = 60.0,
        transport: Optional[httpx.BaseTransport] = None,
    ) -> None:
        self._http = httpx.Client(base_url=base_url, timeout=timeout_s, transport=transport)

    def close(self) -> None:
        """Release the underlying connection pool."""
        self._http.close()

    def health(self) -> HealthStatus:
        """Report readiness; an unreachable API is reported as unhealthy, not raised."""
        try:
            response = self._http.get("/health")
            payload = response.json()
        except (httpx.HTTPError, ValueError):
            return HealthStatus(healthy=False)

        healthy = response.status_code == httpx.codes.OK and bool(payload.get("model_loaded"))
        return HealthStatus(healthy=healthy, model_version=payload.get("model_version"))

    def predict(
        self,
        features: Mapping[str, float],
        product_code: Optional[str] = None,
    ) -> PredictionResult:
        """Classify one product."""
        body: dict[str, Any] = dict(features)
        if product_code:
            body["product_code"] = product_code

        payload = self._post("/v1/predict", body)
        try:
            return PredictionResult(
                nova_group=int(payload["nova_group"]),
                probability=float(payload["probability"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError("The prediction service returned an unexpected response.") from exc

    def predict_batch(
        self,
        records: Sequence[Mapping[str, Any]],
        persist: bool = False,
    ) -> BatchResult:
        """Classify many products in one call."""
        payload = self._post("/v1/predict/batch", {"records": list(records), "persist": persist})
        try:
            return BatchResult(
                model_version=str(payload["model_version"]),
                predictions=tuple(
                    BatchPrediction(
                        index=int(item["index"]),
                        product_code=item.get("product_code"),
                        nova_group=int(item["nova_group"]),
                        probability=float(item["probability"]),
                    )
                    for item in payload["predictions"]
                ),
                errors=tuple(
                    BatchRowFailure(index=int(item["index"]), detail=str(item["detail"]))
                    for item in payload["errors"]
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ApiError("The prediction service returned an unexpected response.") from exc

    def _post(self, path: str, body: Mapping[str, Any]) -> dict[str, Any]:
        try:
            response = self._http.post(path, json=body)
        except httpx.TimeoutException as exc:
            raise ApiConnectionError("The prediction service did not answer in time.") from exc
        except httpx.HTTPError as exc:
            raise ApiConnectionError("The prediction service cannot be reached.") from exc

        self._raise_for_status(response)

        try:
            payload = response.json()
        except ValueError as exc:
            raise ApiError("The prediction service returned invalid JSON.") from exc
        if not isinstance(payload, dict):
            raise ApiError("The prediction service returned an unexpected response.")
        return payload

    @staticmethod
    def _raise_for_status(response: httpx.Response) -> None:
        if response.is_success:
            return
        if response.status_code == httpx.codes.SERVICE_UNAVAILABLE:
            raise ModelUnavailableError("The model is not loaded yet. Try again in a minute.")
        if response.status_code == httpx.codes.UNPROCESSABLE_ENTITY:
            raise ApiError(f"The service rejected the input: {_validation_detail(response)}")
        raise ApiError(f"The prediction service failed (HTTP {response.status_code}).")


def _validation_detail(response: httpx.Response) -> str:
    """Extract readable messages from a FastAPI 422 body."""
    try:
        detail = response.json().get("detail")
    except (ValueError, AttributeError):
        return "invalid request"

    if isinstance(detail, list):
        messages = [
            f"{'.'.join(str(p) for p in item.get('loc', [])[1:])}: {item.get('msg', '')}"
            for item in detail
            if isinstance(item, dict)
        ]
        return "; ".join(messages[:3]) or "invalid request"
    return str(detail) if detail else "invalid request"
