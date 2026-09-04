"""Data ingestion module for Open Food Facts dataset with environment dynamic fallback."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

from open_food_mlops.config.settings import settings

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration class for Data Ingestor settings and feature selection."""

    features: List[str] = field(
        default_factory=lambda: [
            "nova_group",
            "added-sugars_100g",
            "fat_100g",
            "proteins_100g",
            "fruits-vegetables-legumes_100g",
            "sodium_100g",
            "salt_100g",
            "energy-kcal_100g",
            "carbohydrates_100g",
            "water_100g",
        ]
    )
    target: str = "nova_group"
    url: str = field(default_factory=lambda: settings.data_download_url)
    headers: Dict[str, str] = field(
        default_factory=lambda: {"User-Agent": settings.user_agent}
    )
    chunk_size: int = 1024 * 1024
    read_chunk_size: int = 500_000
    data_dir: str = "data"

    def __post_init__(self) -> None:
        """Initialize and validate directory paths."""
        self.base_path = Path(self.data_dir)
        self.raw_dir = self.base_path / "raw"
        self.processed_dir = self.base_path / "processed"

        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        if self.target not in self.features:
            self.features.append(self.target)

    @property
    def raw_file_path(self) -> Path:
        return self.raw_dir / "raw_data.csv.gz"

    @property
    def processed_file_path(self) -> Path:
        return self.processed_dir / "processed_data.parquet"

    @property
    def raw_success_flag(self) -> Path:
        return self.raw_dir / ".success"

    @property
    def processed_success_flag(self) -> Path:
        return self.processed_dir / ".success"


class BaseDataIngestor(ABC):
    """Abstract Base Class defining the interface for data ingestion pipelines."""

    def __init__(self, config: DataConfig) -> None:
        self.config = config

    @abstractmethod
    def download(self) -> None:
        pass

    @abstractmethod
    def process(self) -> None:
        pass

    @abstractmethod
    def run(self) -> pd.DataFrame:
        pass


class OpenFoodFactsDataIngestor(BaseDataIngestor):
    """Concrete Data Ingestor tailored for Open Food Facts dataset."""

    def download(self) -> None:
        """Download dataset in streaming chunks if raw success flag is absent."""
        if (
            self.config.raw_success_flag.exists()
            and self.config.raw_file_path.exists()
        ):
            logger.info("Raw data already exists. Skipping download.")
            return

        logger.info("Downloading raw data from %s...", self.config.url)
        try:
            response = requests.get(
                self.config.url,
                headers=self.config.headers,
                stream=True,
                timeout=60,
            )
            response.raise_for_status()

            with open(self.config.raw_file_path, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=self.config.chunk_size
                ):
                    if chunk:
                        f.write(chunk)

            self.config.raw_success_flag.touch()
            logger.info("Successfully downloaded raw data to %s", self.config.raw_file_path)

        except requests.RequestException as e:
            logger.error("Failed to download raw data: %s", e)
            if self.config.raw_file_path.exists():
                self.config.raw_file_path.unlink()
            raise RuntimeError(f"Data download failed: {e}") from e

    def process(self) -> None:
        """Process raw TSV data in chunks to handle memory limits efficiently."""
        if (
            self.config.processed_success_flag.exists()
            and self.config.processed_file_path.exists()
        ):
            logger.info("Processed data already exists. Skipping processing.")
            return

        if not self.config.raw_file_path.exists():
            raise FileNotFoundError(
                f"Raw data file not found at {self.config.raw_file_path}. Run download() first."
            )

        logger.info("Processing raw data from %s in chunks...", self.config.raw_file_path)

        processed_chunks: List[pd.DataFrame] = []

        def clean_df(df: pd.DataFrame) -> pd.DataFrame:
            for col in ["code", "product_name"]:
                if col in df.columns:
                    df.drop(columns=[col], inplace=True)
            df = df[self.config.features].dropna(subset=[self.config.target]).apply(
                lambda x: pd.to_numeric(x, errors="coerce")
            )
            df = df[df[self.config.target].isin([1.0, 2.0, 3.0, 4.0])]
            df[self.config.target] -= 1
            return df

        try:
            reader = pd.read_csv(
                self.config.raw_file_path,
                sep="\t",
                usecols=lambda col: col in self.config.features,
                chunksize=self.config.read_chunk_size,
                low_memory=False,
                on_bad_lines="skip",
            )

            for chunk in reader:
                if self.config.target not in chunk.columns:
                    raise KeyError(
                        f"Target column '{self.config.target}' not found in raw dataset."
                    )

                cleaned_chunk = clean_df(chunk)
                if not cleaned_chunk.empty:
                    processed_chunks.append(cleaned_chunk)

            if not processed_chunks:
                raise ValueError("Processing resulted in an empty dataset.")

            full_df = pd.concat(processed_chunks, ignore_index=True)
            full_df.to_parquet(
                self.config.processed_file_path, index=False, engine="auto"
            )

            self.config.processed_success_flag.touch()
            logger.info(
                "Successfully saved processed dataset (%d rows) to %s",
                len(full_df),
                self.config.processed_file_path,
            )

        except Exception as e:
            logger.error("Error during data processing: %s", e)
            if self.config.processed_file_path.exists():
                self.config.processed_file_path.unlink()
            raise RuntimeError(f"Data processing failed: {e}") from e

    def run(self) -> pd.DataFrame:
        """Execute full ingestion pipeline safely and return the processed DataFrame."""
        self.download()
        self.process()
        logger.info("Loading dataset from %s", self.config.processed_file_path)
        return pd.read_parquet(self.config.processed_file_path)