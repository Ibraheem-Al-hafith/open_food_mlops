"""Pydantic validation schemas for pipeline configurations."""

from __future__ import annotations

from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field


class DataConfig(BaseModel):
    """Configuration for data providers and splitting strategy."""

    model_config = ConfigDict(extra="forbid")

    provider: str = Field(default="stratified_kfold")
    data_path: str
    target_column: str = "target"
    n_splits: int = Field(default=5, ge=1)
    random_state: int = 42


class FeatureConfig(BaseModel):
    """Configuration for feature engineering pipeline."""

    model_config = ConfigDict(extra="forbid")

    transformers: list[str] = Field(default_factory=list)


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
    """Declarative specification for an end-to-end experiment execution."""

    model_config = ConfigDict(extra="forbid")

    name: str
    seed: int = 42
    data: DataConfig
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    models: list[ModelRunConfig]
    selection: SelectionConfig = Field(default_factory=SelectionConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)