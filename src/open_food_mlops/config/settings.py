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

    # Logging
    log_config_path: str = Field(
        default="config/logging.yaml",
        description="Path to YAML logging config.",
    )

    # MLflow / Tracking
    mlflow_tracking_uri: str = Field(
        default="sqlite:///mlflow.db",
        description="MLflow tracking server URI.",
    )
    mlflow_experiment_name: str = Field(
        default="open-food-mlops-v2",
        description="MLflow experiment namespace.",
    )

    # Serving
    serving_host: str = Field(default="0.0.0.0", description="API host.")
    serving_port: int = Field(default=8000, description="API port.")

    # Data Source Defaults
    data_download_url: str = Field(
        default="https://static.openfoodfacts.org/data/en.openfoodfacts.org.products.csv.gz",
        description="Dataset download URL.",
    )
    user_agent: str = Field(
        default="OpenFoodMLOps/1.0 (contact@example.com)",
        description="HTTP User-Agent header.",
    )


settings = Settings()