"""Evidently data drift report generator enforcing schema alignment between reference and current predictions."""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import pandas as pd
from evidently.presets import DataDriftPreset
from evidently import Report

from open_food_mlops.data.data_ingestor import DataConfig
from open_food_mlops.utils.logger import setup_logging
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

logger = logging.getLogger(__name__)
data_config = DataConfig()
FEATURES = [f for f in data_config.features if f != data_config.target]
PUSHGATEWAY_URL = "localhost:9091"


def push_drift_metrics(
    dataset_drift: float,
    share_drifted: float,
    job_name: str,
    gateway_url: str = PUSHGATEWAY_URL,
) -> None:
    """Push Evidently drift metrics to Prometheus Pushgateway.

    Args:
        dataset_drift: Binary indicator (1.0 for true, 0.0 for false).
        share_drifted: Fraction of total features that drifted [0.0 - 1.0].
        job_name: Job identifier for Prometheus grouping.
        gateway_url: Pushgateway host and port.
    """
    registry = CollectorRegistry()
    Gauge(
        "evidently_dataset_drift",
        "1 if dataset drifted else 0",
        registry=registry,
    ).set(dataset_drift)
    Gauge(
        "evidently_share_of_drifted_columns",
        "Percentage of drifted features",
        registry=registry,
    ).set(share_drifted)

    try:
        push_to_gateway(gateway_url, job=job_name, registry=registry)
        logger.info(
            "Successfully pushed drift metrics to Pushgateway for job: %s",
            job_name,
        )
    except Exception as e:
        logger.warning("Failed to push drift metrics to Pushgateway: %s", e)


def generate_drift_reports(
    reference_path: str = "data/processed/processed_data.parquet",
    current_path: str = "data/production/predictions.parquet",
    output_dir: str = "data/monitoring/reports",
    gateway_url: str = PUSHGATEWAY_URL,
    job_name: str = "batch_evidently_feature_drift",
) -> Path:
    """Generate feature drift report using Evidently and push metrics to Pushgateway.

    Args:
        reference_path: Path to baseline dataset.
        current_path: Path to inference or production predictions dataset.
        output_dir: Destination directory for HTML reports.
        gateway_url: Prometheus Pushgateway endpoint URL.
        job_name: Prometheus Pushgateway batch job identifier.

    Returns:
        Path: Path to saved HTML report.
    """
    ref_file = Path(reference_path)
    curr_file = Path(current_path)

    if not ref_file.exists() or not curr_file.exists():
        raise FileNotFoundError(
            f"Reference file ({reference_path}) or current file ({current_path}) missing."
        )

    ref_df = pd.read_parquet(ref_file)
    curr_df = pd.read_parquet(curr_file)

    feature_cols = [col for col in FEATURES if col in ref_df.columns and col in curr_df.columns]
    reference_features = ref_df[feature_cols]
    current_features = curr_df[feature_cols]

    logger.info("Generating DataDriftPreset report for %d features...", len(feature_cols))
    drift_report = Report(metrics=[DataDriftPreset()])
    result = drift_report.run(reference_data=reference_features, current_data=current_features)

    reports_dir = Path(output_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    today_str = datetime.date.today().isoformat()
    report_file = reports_dir / f"data_drift_{today_str}.html"
    result.save_html(str(report_file))
    logger.info("Data drift report generated successfully: %s", report_file)

    report_dict = result.dict()
    is_drifted = 0.0
    share_drifted = 0.0

    for metric in report_dict.get("metrics", []):
        if metric.get("metric") == "DatasetDriftMetric":
            is_drifted = 1.0 if metric.get("result", {}).get("dataset_drift") else 0.0
            share_drifted = float(metric.get("result", {}).get("share_of_drifted_columns", 0.0))
            break

    push_drift_metrics(
        is_drifted, share_drifted, job_name=job_name, gateway_url=gateway_url
    )

    return report_file

def main() -> None:
    """CLI driver for drift report generation and metric pushing."""
    setup_logging()
    parser = argparse.ArgumentParser(
        description="Generate Evidently Data Drift Report and push metrics to Pushgateway."
    )
    parser.add_argument(
        "--reference-path",
        type=str,
        default="data/monitoring/reference.parquet",
        help="Path to baseline reference dataset.",
    )
    parser.add_argument(
        "--current-path",
        type=str,
        default="data/production/predictions.parquet",
        help="Path to current production predictions dataset.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/monitoring/reports",
        help="Destination directory for HTML reports.",
    )
    parser.add_argument(
        "--gateway-url",
        type=str,
        default=PUSHGATEWAY_URL,
        help="Prometheus Pushgateway endpoint URL.",
    )
    parser.add_argument(
        "--job-name",
        type=str,
        default="batch_evidently_feature_drift",
        help="Prometheus Pushgateway batch job identifier.",
    )

    args = parser.parse_args()

    generate_drift_reports(
        reference_path=args.reference_path,
        current_path=args.current_path,
        output_dir=args.output_dir,
        gateway_url=args.gateway_url,
        job_name=args.job_name,
    )


if __name__ == "__main__":
    main()