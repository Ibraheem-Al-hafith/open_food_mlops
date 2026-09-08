"""Script to extract canonical feature schema and generate a reference dataset for monitoring."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from open_food_mlops.data.data_ingestor import DataConfig
from open_food_mlops.utils.logger import setup_logging

logger = logging.getLogger(__name__)
data_config = DataConfig()
FEATURES = [f for f in data_config.features if f != data_config.target]

def generate_reference_dataset(
    processed_data_path: str = "data/processed/processed_data.parquet",
    output_reference_path: str = "data/monitoring/reference.parquet",
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

    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Processed dataset missing canonical features: {missing}")

    reference_df = df[FEATURES].copy()

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
        default="data/processed/processed_data.parquet",
        help="Path to processed training dataset.",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default="data/monitoring/reference.parquet",
        help="Target path for reference parquet file.",
    )
    args = parser.parse_args()

    generate_reference_dataset(
        processed_data_path=args.processed_path,
        output_reference_path=args.output_path,
    )


if __name__ == "__main__":
    main()