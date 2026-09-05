"""Data ingestion flow CLI pipeline runner."""

from open_food_mlops.data.data_ingestor import OpenFoodFactsDataIngestor, DataConfig
from open_food_mlops.utils.logger import setup_logging
import logging

logger = logging.getLogger(__name__)

if __name__ == "__main__":
    setup_logging()
    logger.info(f"Start Data Ingestion....")
    config = DataConfig()
    ingestor = OpenFoodFactsDataIngestor(config=config)
    _ = ingestor.run()
    logger.info("Data downloaded successfully")

