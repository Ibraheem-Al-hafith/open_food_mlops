"""Evidently prediction drift report generator with Prometheus Pushgateway integration."""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from open_food_mlops.config.settings import settings
from open_food_mlops.utils.logger import setup_logging

logger = logging.getLogger(__name__)

PREDICTION_COLUMN = "prediction"


def push_prediction_drift_metrics(
    prediction_drift: float,
    share_drifted: float,
    job_name: str = "batch_prediction_drift",
    gateway_url: str = settings.pushgateway_url,
) -> None:
    """Push prediction drift metrics to Prometheus Pushgateway."""
    registry = CollectorRegistry()
    Gauge("evidently_prediction_drift", "1 if prediction drift detected else 0", registry=registry).set(prediction_drift)
    Gauge("evidently_prediction_share_drifted", "Share of drifted prediction columns", registry=registry).set(share_drifted)

    try:
        push_to_gateway(gateway_url, job=job_name, registry=registry)
        logger.info("Pushed prediction drift metrics to Pushgateway.")
    except Exception as exc:
        logger.warning("Failed pushing prediction drift metrics: %s", exc)


def generate_prediction_drift_report(
    reference_path: str = str(settings.reference_predictions_path),
    predictions_path: str = str(settings.predictions_path),
    output_dir: str = str(settings.reports_dir),
    gateway_url: str = settings.pushgateway_url,
) -> Path:
    """Compare reference and production prediction distributions and push metrics."""
    ref_path = Path(reference_path)
    pred_path = Path(predictions_path)

    if not ref_path.exists() or not pred_path.exists():
        raise FileNotFoundError(f"Missing prediction datasets: {ref_path} or {pred_path}")

    ref_df = pd.read_parquet(ref_path) if ref_path.suffix == ".parquet" else pd.read_csv(ref_path)

    if pred_path.suffix in [".db", ".sqlite"]:
        from serving.prediction_store import PredictionStore
        pred_df = PredictionStore(db_path=pred_path).read_predictions_dataframe()
    else:
        pred_df = pd.read_parquet(pred_path)

    if PREDICTION_COLUMN not in ref_df.columns or PREDICTION_COLUMN not in pred_df.columns:
        raise ValueError(f"Missing '{PREDICTION_COLUMN}' column in datasets.")

    drift_report = Report(metrics=[DataDriftPreset()])
    result = drift_report.run(
        reference_data=ref_df[[PREDICTION_COLUMN]],
        current_data=pred_df[[PREDICTION_COLUMN]],
    )

    reports_dir = Path(output_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / f"prediction_drift_{datetime.date.today().isoformat()}.html"
    result.save_html(str(report_file))

    report_dict = result.dict()
    is_drifted, share_drifted = 0.0, 0.0

    for metric in report_dict.get("metrics", []):
        if metric.get("metric") == "DatasetDriftMetric":
            is_drifted = 1.0 if metric.get("result", {}).get("dataset_drift") else 0.0
            share_drifted = float(metric.get("result", {}).get("share_of_drifted_columns", 0.0))
            break

    push_prediction_drift_metrics(is_drifted, share_drifted, gateway_url=gateway_url)
    return report_file


if __name__ == "__main__":
    setup_logging()
    generate_prediction_drift_report()