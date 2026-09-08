"""Evidently prediction drift report generator."""

from __future__ import annotations

import argparse
import datetime
import logging
from pathlib import Path

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from open_food_mlops.utils.logger import setup_logging

logger = logging.getLogger(__name__)

PREDICTION_COLUMN = "prediction"


def generate_prediction_drift_report(
    reference_path: str = "data/monitoring/reference_predictions.parquet",
    predictions_path: str = "data/production/predictions.parquet",
    output_dir: str = "data/monitoring/reports",
) -> Path:
    """Compare reference and production prediction distributions.

    Args:
        reference_path: Reference prediction dataset.
        predictions_path: Production prediction dataset.
        output_dir: Directory where the HTML report will be written.

    Returns:
        Path to the generated HTML report.

    Raises:
        FileNotFoundError: If either prediction dataset is missing.
        ValueError: If the prediction column is missing.
    """
    ref_path = Path(reference_path)
    pred_path = Path(predictions_path)

    if not ref_path.exists():
        raise FileNotFoundError(
            f"Reference predictions dataset not found at {reference_path}"
        )

    if not pred_path.exists():
        raise FileNotFoundError(
            f"Production predictions store not found at {predictions_path}"
        )

    logger.info("Reading reference predictions from %s", reference_path)
    ref_df = pd.read_parquet(ref_path)

    logger.info("Reading production predictions from %s", predictions_path)
    pred_df = pd.read_parquet(predictions_path)

    if PREDICTION_COLUMN not in ref_df.columns:
        raise ValueError(
            f"Missing '{PREDICTION_COLUMN}' column in reference predictions."
        )

    if PREDICTION_COLUMN not in pred_df.columns:
        raise ValueError(
            f"Missing '{PREDICTION_COLUMN}' column in production predictions."
        )

    reference_predictions = ref_df[[PREDICTION_COLUMN]]
    current_predictions = pred_df[[PREDICTION_COLUMN]]

    logger.info(
        "Generating prediction drift report using %d production predictions...",
        len(current_predictions),
    )

    drift_report = Report(
        metrics=[DataDriftPreset()]
    )

    result = drift_report.run(
        reference_data=reference_predictions,
        current_data=current_predictions,
    )

    reports_dir = Path(output_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    today_str = datetime.date.today().isoformat()
    report_file = reports_dir / f"prediction_drift_{today_str}.html"

    result.save_html(str(report_file))

    logger.info(
        "Prediction drift report generated successfully: %s",
        report_file,
    )

    return report_file


def main() -> None:
    """CLI driver for prediction drift report generation."""
    setup_logging()

    parser = argparse.ArgumentParser(
        description="Generate Evidently Prediction Drift Report"
    )

    parser.add_argument(
        "--reference-path",
        type=str,
        default="data/monitoring/reference_predictions.parquet",
        help="Path to reference predictions parquet.",
    )

    parser.add_argument(
        "--predictions-path",
        type=str,
        default="data/production/predictions.parquet",
        help="Path to production predictions parquet.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/monitoring/reports",
        help="Output directory for generated HTML reports.",
    )

    args = parser.parse_args()

    generate_prediction_drift_report(
        reference_path=args.reference_path,
        predictions_path=args.predictions_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()