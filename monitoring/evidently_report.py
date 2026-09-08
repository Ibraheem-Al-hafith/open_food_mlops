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

logger = logging.getLogger(__name__)
data_config = DataConfig()
FEATURES = [f for f in data_config.features if f != data_config.target]


def generate_drift_report(
    reference_path: str = "data/monitoring/reference.parquet",
    predictions_path: str = "data/production/predictions.parquet",
    output_dir: str = "data/monitoring/reports",
) -> Path:
    """Compare baseline reference data with production predictions and output an HTML drift report.

    Args:
        reference_path: Path to canonical baseline reference dataset.
        predictions_path: Path to target production prediction store.
        output_dir: Target directory for storing HTML drift reports.

    Returns:
        Path object pointing to the generated HTML report file.

    Raises:
        FileNotFoundError: If reference or predictions parquet datasets are missing.
        ValueError: If required canonical features are absent in either dataset.
    """
    ref_path = Path(reference_path)
    pred_path = Path(predictions_path)

    if not ref_path.exists():
        raise FileNotFoundError(f"Reference dataset not found at {reference_path}")
    if not pred_path.exists():
        raise FileNotFoundError(f"Predictions store not found at {predictions_path}")

    logger.info("Reading reference dataset from %s", reference_path)
    ref_df = pd.read_parquet(ref_path)

    logger.info("Reading production predictions from %s", predictions_path)
    pred_df = pd.read_parquet(predictions_path)

    missing_ref = [f for f in FEATURES if f not in ref_df.columns]
    missing_pred = [f for f in FEATURES if f not in pred_df.columns]

    if missing_ref or missing_pred:
        raise ValueError(
            f"Schema mismatch! Missing in reference: {missing_ref}, Missing in production: {missing_pred}"
        )

    reference_features = ref_df[FEATURES]
    current_features = pred_df[FEATURES]

    logger.info("Generating DataDriftPreset report for %d features...", len(FEATURES))
    
    # In Evidently 0.7+, metrics/presets are passed as instances to the Report metric list
    drift_report = Report(metrics=[DataDriftPreset()])
    result = drift_report.run(reference_data=reference_features, current_data=current_features)

    reports_dir = Path(output_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    today_str = datetime.date.today().isoformat()
    report_file = reports_dir / f"data_drift_{today_str}.html"

    result.save_html(str(report_file))
    logger.info("Data drift report generated successfully: %s", report_file)
    return report_file


def main() -> None:
    """CLI driver for drift report generation."""
    setup_logging()
    parser = argparse.ArgumentParser(description="Generate Evidently Data Drift Report")
    parser.add_argument(
        "--reference-path",
        type=str,
        default="data/monitoring/reference.parquet",
        help="Path to monitoring baseline reference parquet.",
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

    generate_drift_report(
        reference_path=args.reference_path,
        predictions_path=args.predictions_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()