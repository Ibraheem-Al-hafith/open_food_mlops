"""Script to extract canonical feature schema and generate a reference dataset for monitoring."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from open_food_mlops.utils.logger import setup_logging
from open_food_mlops.config.features import FEATURE_COLUMNS
from open_food_mlops.config.settings import settings


logger = logging.getLogger(__name__)


def generate_reference_dataset(
    processed_data_path: str = str(settings.ground_truth_path),
    output_reference_path: str = str(settings.reference_data_path),
) -> None:
    """Extract canonical feature schema from training/processed dataset and save to reference store.

    Args:
        processed_data_path: Source path for baseline training/processed dataset.
        output_reference_path: Target path for reference dataset.
    """
    logger.info("Loading processed dataset from %s...", processed_data_path)
    source_path = Path(processed_data_path)
    if not source_path.exists():
        raise FileNotFoundError(f"Source file not found at {processed_data_path}")

    if processed_data_path.endswith(".parquet"):
        df = pd.read_parquet(processed_data_path)
    else:
        df = pd.read_csv(processed_data_path)

    missing = [f for f in FEATURE_COLUMNS if f not in df.columns]
    if missing:
        raise ValueError(f"Processed dataset missing canonical features: {missing}")

    reference_df = df[FEATURE_COLUMNS].copy()

    output_path = Path(output_reference_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    reference_df.to_parquet(output_path, index=False)
    logger.info("Successfully exported reference dataset with shape %s to %s", reference_df.shape, output_reference_path)


def main() -> None:
    """CLI driver for reference dataset creation."""
    setup_logging()
    parser = argparse.ArgumentParser(description="Create Reference Monitoring Dataset")
    parser.add_argument(
        "--processed-path",
        type=str,
        default=str(settings.ground_truth_path),
        help="Path to processed training dataset.",
    )

    parser.add_argument(
        "--output-path",
        type=str,
        default=str(settings.reference_data_path),
        help="Target path for reference parquet file.",
    )
    args = parser.parse_args()

    generate_reference_dataset(
        processed_data_path=args.processed_path,
        output_reference_path=args.output_path,
    )


if __name__ == "__main__":
    main()