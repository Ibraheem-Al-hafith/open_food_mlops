"""Prefect flow for building monitoring reference data."""

from __future__ import annotations

import logging
from pathlib import Path

from prefect import flow, task

from open_food_mlops.config.settings import settings
from open_food_mlops.data.data_ingestor import (
    DataConfig,
    OpenFoodFactsDataIngestor,
)
from open_food_mlops.utils.logger import setup_logging
from pipelines.create_reference_data import generate_reference_dataset
from pipelines.create_reference_predictions import generate_reference_predictions

logger = logging.getLogger(__name__)


@task(
    name="Ingest Open Food Facts Data",
    retries=2,
    retry_delay_seconds=60,
    log_prints=True,
)
def ingest_data(refresh_source: bool = False) -> None:
    config = DataConfig()

    if refresh_source:
        logger.info("Refreshing source data.")

        paths = [
            config.raw_success_flag,
            config.processed_success_flag,
            config.raw_file_path,
            config.processed_file_path,
        ]

        for path in paths:
            if path.exists():
                path.unlink()

    ingestor = OpenFoodFactsDataIngestor(config=config)
    ingestor.run()

    logger.info("Data ingestion completed.")


@task(
    name="Generate Reference Dataset",
    retries=1,
    log_prints=True,
)
def generate_reference() -> None:
    generate_reference_dataset(
        processed_data_path=str(settings.ground_truth_path),
        output_reference_path=str(settings.reference_data_path),
    )

    logger.info(
        "Reference dataset generated at %s",
        settings.reference_data_path,
    )


@task(
    name="Generate Reference Predictions",
    retries=1,
    log_prints=True,
)
def generate_reference_prediction_data() -> None:
    generate_reference_predictions()

    logger.info(
        "Reference predictions generated at %s",
        settings.reference_predictions_path,
    )


@flow(
    name="open-food-mlops-reference",
    log_prints=True,
)
def reference_pipeline(refresh_source: bool = False) -> None:
    """
    Build the monitoring baseline from source data.

    refresh_source=False:
        Reuse existing downloaded/processed data when valid.

    refresh_source=True:
        Download and process the source dataset again.
    """
    setup_logging()

    print("=" * 60)
    print("BUILDING OPEN FOOD MLOPS MONITORING REFERENCE")
    print("=" * 60)

    ingest_data(refresh_source=refresh_source)
    generate_reference()
    generate_reference_prediction_data()

    print("=" * 60)
    print("REFERENCE BUILD COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    reference_pipeline()
