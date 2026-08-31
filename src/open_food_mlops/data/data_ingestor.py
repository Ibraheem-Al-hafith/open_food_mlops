"""Data ingestion module for Open Food Facts dataset.

This module provides an extensible framework and a concrete implementation
for downloading, caching, and preprocessing tabular datasets for ML pipelines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd
import requests

logger = logging.getLogger(__name__)


@dataclass
class DataConfig:
    """Configuration class for Data Ingestor settings and feature selection.

    Attributes:
        features: List of feature column names to extract from the raw dataset.
        target: Target column name for the machine learning task.
        url: Direct download URL for the dataset.
        headers: HTTP headers to include with the download request.
        chunk_size: Stream download chunk size in bytes (e.g., 1024 * 1024 for 1MB chunks).
        read_chunk_size: Row batch size for chunked pandas CSV processing.
        data_dir: Base directory where raw and processed data will be saved.
    """

    features: List[str] = field(
        default_factory=lambda: [
            "nova_group",
            "additives_n",
            "code",
            "product_name",
            "energy-kj_100g",
            "energy-kcal_100g",
            "energy_100g",
            "energy-from-fat_100g",
            "fat_100g",
            "saturated-fat_100g",
            "butyric-acid_100g",
            "caproic-acid_100g",
            "caprylic-acid_100g",
            "capric-acid_100g",
            "lauric-acid_100g",
            "myristic-acid_100g",
            "palmitic-acid_100g",
            "stearic-acid_100g",
            "arachidic-acid_100g",
            "behenic-acid_100g",
            "lignoceric-acid_100g",
            "cerotic-acid_100g",
            "montanic-acid_100g",
            "melissic-acid_100g",
            "unsaturated-fat_100g",
            "monounsaturated-fat_100g",
            "omega-9-fat_100g",
            "polyunsaturated-fat_100g",
            "omega-3-fat_100g",
            "omega-6-fat_100g",
            "alpha-linolenic-acid_100g",
            "eicosapentaenoic-acid_100g",
            "docosahexaenoic-acid_100g",
            "linoleic-acid_100g",
            "arachidonic-acid_100g",
            "gamma-linolenic-acid_100g",
            "dihomo-gamma-linolenic-acid_100g",
            "oleic-acid_100g",
            "elaidic-acid_100g",
            "gondoic-acid_100g",
            "mead-acid_100g",
            "erucic-acid_100g",
            "nervonic-acid_100g",
            "trans-fat_100g",
            "cholesterol_100g",
            "carbohydrates_100g",
            "sugars_100g",
            "added-sugars_100g",
            "sucrose_100g",
            "glucose_100g",
            "fructose_100g",
            "galactose_100g",
            "lactose_100g",
            "maltose_100g",
            "maltodextrins_100g",
            "psicose_100g",
            "starch_100g",
            "polyols_100g",
            "erythritol_100g",
            "isomalt_100g",
            "maltitol_100g",
            "sorbitol_100g",
            "fiber_100g",
            "soluble-fiber_100g",
            "polydextrose_100g",
            "insoluble-fiber_100g",
            "proteins_100g",
            "casein_100g",
            "serum-proteins_100g",
            "nucleotides_100g",
            "salt_100g",
            "added-salt_100g",
            "sodium_100g",
            "alcohol_100g",
            "vitamin-a_100g",
            "beta-carotene_100g",
            "vitamin-d_100g",
            "vitamin-e_100g",
            "vitamin-k_100g",
            "vitamin-c_100g",
            "vitamin-b1_100g",
            "vitamin-b2_100g",
            "vitamin-pp_100g",
            "vitamin-b6_100g",
            "vitamin-b9_100g",
            "folates_100g",
            "vitamin-b12_100g",
            "biotin_100g",
            "pantothenic-acid_100g",
            "silica_100g",
            "bicarbonate_100g",
            "potassium_100g",
            "chloride_100g",
            "calcium_100g",
            "phosphorus_100g",
            "iron_100g",
            "magnesium_100g",
            "zinc_100g",
            "copper_100g",
            "manganese_100g",
            "fluoride_100g",
            "selenium_100g",
            "chromium_100g",
            "molybdenum_100g",
            "iodine_100g",
            "caffeine_100g",
            "taurine_100g",
            "methylsulfonylmethane_100g",
            "hydroxymethylbutyrate_100g",
            "ph_100g",
            "fruits-vegetables-legumes_100g",
            "collagen-meat-protein-ratio_100g",
            "cocoa_100g",
            "chlorophyl_100g",
            "carbon-footprint_100g",
            "glycemic-index_100g",
            "water-hardness_100g",
            "choline_100g",
            "phylloquinone_100g",
            "beta-glucan_100g",
            "inositol_100g",
            "carnitine_100g",
            "sulphate_100g",
            "nitrate_100g",
            "acidity_100g",
            "carbohydrates-total_100g",
            "water_100g",
        ]
    )
    target: str = "nova_group"
    url: str = (
        "https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz"
    )
    headers: Dict[str, str] = field(
        default_factory=lambda: {
            "User-Agent": "MySimpleMLProject/1.0 (contact@example.com)"
        }
    )
    chunk_size: int = 1024 * 1024  # 1 MB chunk download size
    read_chunk_size: int = 500_000  # DataFrame reader chunk size
    data_dir: str = "data"

    def __post_init__(self) -> None:
        """Initialize and validate directory paths."""
        self.base_path = Path(self.data_dir)
        self.raw_dir = self.base_path / "raw"
        self.processed_dir = self.base_path / "processed"

        # Ensure directory infrastructure exists
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

        # Ensure target is present in the extracted features
        if self.target not in self.features:
            self.features.append(self.target)

    @property
    def raw_file_path(self) -> Path:
        """Path to raw downloaded compressed file."""
        return self.raw_dir / "raw_data.csv.gz"

    @property
    def processed_file_path(self) -> Path:
        """Path to processed Parquet dataset."""
        return self.processed_dir / "processed_data.parquet"

    @property
    def raw_success_flag(self) -> Path:
        """Sentinel flag indicating raw download completion."""
        return self.raw_dir / ".success"

    @property
    def processed_success_flag(self) -> Path:
        """Sentinel flag indicating data processing completion."""
        return self.processed_dir / ".success"


class BaseDataIngestor(ABC):
    """Abstract Base Class defining the interface for data ingestion pipelines."""

    def __init__(self, config: DataConfig) -> None:
        """Initialize the data ingestor with a configuration instance.

        Args:
            config: Configuration settings object.
        """
        self.config = config

    @abstractmethod
    def download(self) -> None:
        """Download raw data from source location."""
        pass

    @abstractmethod
    def process(self) -> None:
        """Process raw data, filter columns, handle missing values, and persist clean output."""
        pass

    @abstractmethod
    def run(self) -> pd.DataFrame:
        """Execute the ingestion workflow and return a pandas DataFrame.

        Returns:
            pd.DataFrame: Cleaned and processed dataset ready for training/inference.
        """
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

        logger.info(f"Downloading raw data from {self.config.url}...")
        try:
            response = requests.get(
                self.config.url,
                headers=self.config.headers,
                stream=True,
                timeout=60,
            )
            response.raise_for_status()

            # Stream response content to file to minimize RAM usage
            with open(self.config.raw_file_path, "wb") as f:
                for chunk in response.iter_content(
                    chunk_size=self.config.chunk_size
                ):
                    if chunk:
                        f.write(chunk)

            # Touch raw success file
            self.config.raw_success_flag.touch()
            logger.info(
                f"Successfully downloaded raw data to {self.config.raw_file_path}"
            )

        except requests.RequestException as e:
            logger.error(f"Failed to download raw data: {e}")
            if self.config.raw_file_path.exists():
                self.config.raw_file_path.unlink()  # Clean up partial download
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

        logger.info(
            f"Processing raw data from {self.config.raw_file_path} in chunks..."
        )

        processed_chunks: List[pd.DataFrame] = []
        def clean_df(df: pd.DataFrame, feature_columns: List[str] = self.config.features) -> pd.DataFrame:
            """
            Helper function to clean the data frame, it drop non-convertable str to int rows
            Args:
                df: pd.DataFrame
                feature_columns: List[str]
            Returns:
                pd.DataFrame
            """
            df = df[feature_columns].dropna(subset=[self.config.target]).apply(lambda x:pd.to_numeric(x, errors="coerce"))
            return df

        try:
            # Note: Open Food Facts CSV downloads are tab-separated (\t)
            reader = pd.read_csv(
                self.config.raw_file_path,
                sep="\t",
                usecols=lambda col: col in self.config.features,
                chunksize=self.config.read_chunk_size,
                low_memory=False,
                on_bad_lines="skip",
            )

            for chunk_idx, chunk in enumerate(reader):
                # Ensure target column is present in chunk
                if self.config.target not in chunk.columns:
                    raise KeyError(
                        f"Target column '{self.config.target}' not found in raw dataset."
                    )

                # Basic Cleaning: Drop rows where target column is NaN
                cleaned_chunk = clean_df(chunk)

                if not cleaned_chunk.empty:
                    processed_chunks.append(cleaned_chunk)

            if not processed_chunks:
                raise ValueError("Processing resulted in an empty dataset.")

            # Combine chunks and save to Parquet
            full_df = pd.concat(processed_chunks, ignore_index=True)
            full_df.to_parquet(
                self.config.processed_file_path, index=False, engine="auto"
            )

            # Touch processed success file
            self.config.processed_success_flag.touch()
            logger.info(
                f"Successfully saved processed dataset ({len(full_df)} rows) to "
                f"{self.config.processed_file_path}"
            )

        except Exception as e:
            logger.error(f"Error during data processing: {e}")
            if self.config.processed_file_path.exists():
                self.config.processed_file_path.unlink()
            raise RuntimeError(f"Data processing failed: {e}") from e

    def run(self) -> pd.DataFrame:
        """Execute full ingestion pipeline safely and return the processed DataFrame."""
        self.download()
        self.process()

        logger.info(f"Loading dataset from {self.config.processed_file_path}")
        return pd.read_parquet(self.config.processed_file_path)


# Example usage
if __name__ == "__main__":
    config = DataConfig()
    ingestor = OpenFoodFactsDataIngestor(config=config)
    df = ingestor.run()
    logger.info(f"Loaded DataFrame shape: {df.shape}")
    print(df.head())