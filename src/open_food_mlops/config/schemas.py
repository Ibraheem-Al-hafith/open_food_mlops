"""Pydantic validation schemas for pipeline configurations."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    """Configuration for dataset ingestion, sampling, and splitting."""

    model_config = ConfigDict(extra="forbid")

    data_path: str
    target_column: str = "nova_group"
    sample_fraction: float = Field(default=1.0, gt=0.0, le=1.0)
    test_size: float = Field(default=0.2, gt=0.0, lt=1.0)
    n_splits: int = Field(default=5, ge=2)
    validation_method: Literal[
        "train_test_split", "kfold", "stratified_kfold"
    ] = "stratified_kfold"
    random_state: int = 42


class FeatureConfig(BaseModel):
    """Configuration for feature engineering pipeline."""

    model_config = ConfigDict(extra="forbid")

    transformers: list[str] = Field(default_factory=lambda: ["identity"])


class TuningConfig(BaseModel):
    """Hyperparameter optimization configuration."""

    model_config = ConfigDict(extra="forbid")

    enabled: bool = False
    method: str = "optuna"
    trials: int = Field(default=20, gt=0)
    direction: Literal["maximize", "minimize"] = "maximize"
    random_state: int = 42


class ModelRunConfig(BaseModel):
    """Configuration for an individual model run within an experiment."""

    model_config = ConfigDict(extra="forbid")

    name: str
    enabled: bool = True
    params: dict[str, Any] = Field(default_factory=dict)
    tuning: TuningConfig = Field(default_factory=TuningConfig)


class SelectionConfig(BaseModel):
    """Model selection and quality gate thresholds."""

    model_config = ConfigDict(extra="forbid")

    primary_metric: str = "macro_f1"
    direction: Literal["maximize", "minimize"] = "maximize"
    gates: dict[str, float] = Field(default_factory=dict)


class TrackingConfig(BaseModel):
    """Experiment tracking configuration."""

    model_config = ConfigDict(extra="forbid")

    backend: str = "mlflow"
    tracking_uri: str = "sqlite:///mlflow.db"
    experiment_name: str = "open-food-mlops"


class ExperimentPlan(BaseModel):
    """Declarative specification for end-to-end pipeline execution."""

    model_config = ConfigDict(extra="forbid")

    name: str
    seed: int = 42
    data: DataConfig
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    models: list[ModelRunConfig]
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)