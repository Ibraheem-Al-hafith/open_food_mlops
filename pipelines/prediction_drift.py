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
    """Push prediction drift metrics to Prometheus Pushgateway.

    Args:
        prediction_drift: Flag indicating if drift was detected (1.0) or not (0.0).
        share_drifted: Proportion of drifted prediction columns in [0.0, 1.0].
        job_name: Name of the job tag sent to Pushgateway.
        gateway_url: Destination URL for Prometheus Pushgateway.
    """
    registry = CollectorRegistry()
    Gauge(
        "evidently_prediction_drift",
        "1 if prediction drift detected else 0",
        registry=registry,
    ).set(prediction_drift)
    Gauge(
        "evidently_prediction_share_drifted",
        "Share of drifted prediction columns",
        registry=registry,
    ).set(share_drifted)

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
    """Compare reference and production prediction distributions and push metrics.

    Args:
        reference_path: Path to reference dataset file.
        predictions_path: Path to current predictions dataset or DB file.
        output_dir: Directory where the output HTML report will be written.
        gateway_url: Destination URL for Prometheus Pushgateway.

    Returns:
        Path object pointing to the written HTML report.

    Raises:
        FileNotFoundError: If input reference or prediction file does not exist.
        ValueError: If prediction column is missing or calculated metrics are invalid.
        RuntimeError: If Evidently metrics structure is missing or malformed.
    """
    ref_path = Path(reference_path)
    pred_path = Path(predictions_path)

    if not ref_path.exists() or not pred_path.exists():
        raise FileNotFoundError(
            f"Missing prediction datasets: {ref_path} or {pred_path}"
        )

    ref_df = (
        pd.read_parquet(ref_path)
        if ref_path.suffix == ".parquet"
        else pd.read_csv(ref_path)
    )

    if pred_path.suffix in [".db", ".sqlite"]:
        from serving.prediction_store import PredictionStore

        pred_df = PredictionStore(db_path=pred_path).read_predictions_dataframe()
    else:
        pred_df = pd.read_parquet(pred_path)

    if (
        PREDICTION_COLUMN not in ref_df.columns
        or PREDICTION_COLUMN not in pred_df.columns
    ):
        raise ValueError(f"Missing '{PREDICTION_COLUMN}' column in datasets.")

    drift_report = Report(metrics=[DataDriftPreset()])
    result = drift_report.run(
        reference_data=ref_df[[PREDICTION_COLUMN]],
        current_data=pred_df[[PREDICTION_COLUMN]],
    )

    reports_dir = Path(output_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_file = (
        reports_dir / f"prediction_drift_{datetime.date.today().isoformat()}.html"
    )
    result.save_html(str(report_file))

    report_dict = result.dict()

    prediction_drift_metric = next(
        (
            metric
            for metric in report_dict.get("metrics", [])
            if metric.get("metric_name", "").startswith(
                f"ValueDrift(column={PREDICTION_COLUMN},"
            )
        ),
        None,
    )

    if prediction_drift_metric is None:
        raise RuntimeError(
            f"Evidently report did not contain ValueDrift for "
            f"'{PREDICTION_COLUMN}'."
        )

    if "value" not in prediction_drift_metric:
        raise RuntimeError(
            "Evidently ValueDrift metric is missing its 'value'."
        )

    prediction_drift_distance = float(prediction_drift_metric["value"])

    if not 0.0 <= prediction_drift_distance <= 1.0:
        raise ValueError(
            f"Invalid prediction drift distance: {prediction_drift_distance}. "
            "Expected a value between 0 and 1."
        )

    # The ValueDrift metric uses a threshold of 0.1.
    is_drifted = 1.0 if prediction_drift_distance >= 0.1 else 0.0
    share_drifted = is_drifted

    if not 0.0 <= share_drifted <= 1.0:
        raise ValueError(
            f"Invalid share_of_drifted_columns: {share_drifted}. "
            "Expected a value between 0 and 1."
        )

    push_prediction_drift_metrics(is_drifted, share_drifted, gateway_url=gateway_url)
    return report_file


if __name__ == "__main__":
    setup_logging()
    generate_prediction_drift_report()