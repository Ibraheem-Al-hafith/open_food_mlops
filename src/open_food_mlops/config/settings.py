"""Centralized application settings loaded dynamically from environment variables or .env."""

from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Global application settings schema supporting env overrides."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
    )

    log_config_path: str = Field(
        default="config/logging.yaml",
        description="Path to YAML logging config.",
    )

    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlflow.db",
        description="MLflow tracking server URI.",
    )
    mlflow_experiment_name: str = Field(
        default="open-food-mlops-v2",
        description="MLflow experiment namespace.",
    )

    serving_host: str = Field(default="0.0.0.0", description="API host.")
    serving_port: int = Field(default=8000, description="API port.")

    pushgateway_url: str = Field(
        default="http://pushgateway:9091",
        description="Prometheus Pushgateway URL.",
    )

    data_download_url: str = Field(
        default="https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz",
        description="Dataset download URL.",
    )
    user_agent: str = Field(
        default="OpenFoodMLOps/1.0 (contact@example.com)",
        description="HTTP User-Agent header.",
    )

    predictions_path: Path = Field(
        default_factory=lambda: Path("data/production/predictions.db")
    )
    ground_truth_path: Path = Field(
        default_factory=lambda: Path("data/processed/processed_data.parquet")
    )
    reference_data_path: Path = Field(
        default_factory=lambda: Path("data/monitoring/reference.parquet")
    )
    reference_predictions_path: Path = Field(
        default_factory=lambda: Path("data/monitoring/reference_predictions.parquet")
    )
    reports_dir: Path = Field(
        default_factory=lambda: Path("data/monitoring/reports")
    )


settings = Settings()