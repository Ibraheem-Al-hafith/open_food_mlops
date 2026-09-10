"""Centralized application settings loaded dynamically from environment variables or .env."""

from pathlib import Path
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    base_dir: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parents[3]
    )

    app_env: str = Field(
        default="development",
        description="Application environment.",
    )

    log_config_path: Path = Field(
        default=Path("config/logging.yaml"),
        description="Path to YAML logging config.",
    )

    mlflow_tracking_uri: str = Field(
        default="http://localhost:5000",
        description="MLflow tracking server URI.",
    )

    mlflow_experiment_name: str = Field(
        default="open-food-mlops-v2",
        description="Default MLflow experiment name.",
    )

    serving_host: str = Field(
        default="0.0.0.0",
        description="API bind host.",
    )

    serving_port: int = Field(
        default=8000,
        description="API bind port.",
    )

    pushgateway_url: str = Field(
        default="http://localhost:9091",
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
        default=Path("data/production/predictions.db")
    )

    ground_truth_path: Path = Field(
        default=Path("data/processed/processed_data.parquet")
    )

    reference_data_path: Path = Field(
        default=Path("data/monitoring/reference.parquet")
    )

    reference_predictions_path: Path = Field(
        default=Path("data/monitoring/reference_predictions.parquet")
    )

    reports_dir: Path = Field(
        default=Path("data/monitoring/reports")
    )
    
    @model_validator(mode="after")
    def resolve_absolute_paths(self) -> "Settings":
        # Ensure base_dir is fully resolved and absolute
        object.__setattr__(self, "base_dir", self.base_dir.resolve())

        path_fields = (
            "log_config_path",
            "predictions_path",
            "ground_truth_path",
            "reference_data_path",
            "reference_predictions_path",
            "reports_dir",
        )

        for field_name in path_fields:
            current_path: Path = getattr(self, field_name)
            if not current_path.is_absolute():
                # Use object.__setattr__ to bypass frozen/validation state cleanly
                object.__setattr__(self, field_name, (self.base_dir / current_path).resolve())

        return self
    @model_validator(mode="after")
    def validate_production_settings(self):
        if self.app_env == "production":
            if "localhost" in self.mlflow_tracking_uri or "127.0.0.1" in self.mlflow_tracking_uri:
                raise ValueError(
                    "MLFLOW_TRACKING_URI cannot point to localhost in production."
                )

            if "localhost" in self.pushgateway_url or "127.0.0.1" in self.pushgateway_url:
                raise ValueError(
                    "PUSHGATEWAY_URL cannot point to localhost in production."
                )

        return self


settings = Settings()