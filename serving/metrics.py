"""Prometheus metrics middleware and instrumentation for serving app."""

from __future__ import annotations

import time
from typing import Callable

from fastapi import FastAPI, Request, Response
from prometheus_client import Gauge, Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests received",
    ["method", "endpoint", "status_code"],
)

HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds",
    ["method", "endpoint"],
)

PREDICTION_COUNTER = Counter(
    "model_predictions_total",
    "Total model predictions served",
    ["nova_group"],
)

PREDICTION_LATENCY = Histogram(
    "model_prediction_duration_seconds",
    "Time spent running model prediction logic in seconds",
)

MODEL_PRODUCTION_MACRO_F1 = Gauge(
    "model_production_macro_f1",
    "Calculated production Macro-F1 score against delayed ground truth labels",
)

MODEL_DEGRADATION_RATIO = Gauge(
    "model_degradation_ratio",
    "Ratio of current production Macro-F1 over baseline champion training Macro-F1",
)


def setup_monitoring(app: FastAPI) -> None:
    """Instrument FastAPI application with Prometheus metrics middleware and endpoint."""

    @app.middleware("http")
    async def metrics_middleware(request: Request, call_next: Callable) -> Response:
        start_time = time.perf_counter()
        response = await call_next(request)
        duration = time.perf_counter() - start_time

        endpoint = request.url.path
        method = request.method
        status_code = str(response.status_code)

        if endpoint != "/metrics":
            HTTP_REQUESTS_TOTAL.labels(
                method=method,
                endpoint=endpoint,
                status_code=status_code,
            ).inc()

            HTTP_REQUEST_DURATION_SECONDS.labels(
                method=method,
                endpoint=endpoint,
            ).observe(duration)

        return response

    @app.get("/metrics", include_in_schema=False)
    def metrics() -> Response:
        """Prometheus metrics scraping endpoint."""
        return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)