"""Evidently data drift report generator enforcing schema alignment."""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from open_food_mlops.config.features import FEATURE_COLUMNS
from open_food_mlops.config.settings import settings
from open_food_mlops.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def push_drift_metrics(
    dataset_drift: float,
    share_drifted: float,
    job_name: str,
    gateway_url: str = settings.pushgateway_url,
) -> None:
    """Push Evidently drift metrics to Prometheus Pushgateway."""
    registry = CollectorRegistry()
    Gauge("evidently_dataset_drift", "1 if dataset drifted else 0", registry=registry).set(dataset_drift)
    Gauge("evidently_share_of_drifted_columns", "Percentage of drifted features", registry=registry).set(share_drifted)

    try:
        push_to_gateway(gateway_url, job=job_name, registry=registry)
        logger.info("Pushed drift metrics to Pushgateway for job: %s", job_name)
    except Exception as e:
        logger.warning("Failed to push drift metrics to Pushgateway: %s", e)


def generate_drift_reports(
    reference_path: str = str(settings.reference_data_path),
    current_path: str = str(settings.predictions_path),
    output_dir: str = str(settings.reports_dir),
    gateway_url: str = settings.pushgateway_url,
    job_name: str = "batch_evidently_feature_drift",
) -> Path:
    """Generate feature drift report using Evidently and push metrics."""
    ref_file = Path(reference_path)
    curr_file = Path(current_path)

    if not ref_file.exists() or not curr_file.exists():
        raise FileNotFoundError(f"Reference ({ref_file}) or Current ({curr_file}) data missing.")

    ref_df = pd.read_parquet(ref_file)
    curr_df = pd.read_parquet(curr_file)

    feature_cols = [col for col in FEATURE_COLUMNS if col in ref_df.columns and col in curr_df.columns]
    
    drift_report = Report(metrics=[DataDriftPreset()])
    result = drift_report.run(reference_data=ref_df[feature_cols], current_data=curr_df[feature_cols])

    reports_dir = Path(output_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = reports_dir / f"data_drift_{datetime.date.today().isoformat()}.html"
    result.save_html(str(report_file))

    report_dict = result.dict()
    is_drifted, share_drifted = 0.0, 0.0

    for metric in report_dict.get("metrics", []):
        if metric.get("metric") == "DatasetDriftMetric":
            is_drifted = 1.0 if metric.get("result", {}).get("dataset_drift") else 0.0
            share_drifted = float(metric.get("result", {}).get("share_of_drifted_columns", 0.0))
            break

    push_drift_metrics(is_drifted, share_drifted, job_name=job_name, gateway_url=gateway_url)
    return report_file


if __name__ == "__main__":
    setup_logging()
    generate_drift_reports()